"""
CineBrain Cast & Crew Management
Comprehensive handling of cast, crew, and credits data
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy import and_, or_, func
from flask import has_app_context

from .slug import SlugManager
from .errors import DetailsError, handle_api_error

logger = logging.getLogger(__name__)

class CastCrewManager:
    """Manages cast and crew data for content"""
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models.get('Content')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        
        logger.info("CastCrewManager initialized successfully")
    
    def get_cast_crew(self, content_id: int) -> Dict:
        """Get comprehensive cast and crew data"""
        try:
            if not has_app_context():
                logger.warning("No app context for cast/crew retrieval")
                return self._empty_cast_crew()
            
            content = self.Content.query.get(content_id) if content_id else None
            if not content:
                logger.warning(f"Content not found for ID: {content_id}")
                return self._empty_cast_crew()
            
            # Try to get from database first
            cast_crew = self._get_cast_crew_from_db(content_id)
            
            # Check if we have sufficient data
            total_cast_crew = len(cast_crew['cast']) + sum(len(crew_list) for crew_list in cast_crew['crew'].values())
            
            if total_cast_crew >= 5:
                logger.info(f"Found {total_cast_crew} cast/crew members in database for content {content_id}")
                return cast_crew
            
            # Fetch from external APIs if insufficient data
            if content.tmdb_id:
                logger.info(f"Fetching comprehensive cast/crew from TMDB for content {content_id}")
                external_cast_crew = self._fetch_and_save_all_cast_crew(content)
                if external_cast_crew:
                    return external_cast_crew
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew for content {content_id}: {e}")
            return self._empty_cast_crew()
    
    def _get_cast_crew_from_db(self, content_id: int) -> Dict:
        """Retrieve cast and crew from database"""
        cast_crew = self._empty_cast_crew()
        
        try:
            # Get cast members
            cast_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'cast'
            ).order_by(
                self.ContentPerson.order.asc().nullslast()
            ).limit(25).all()  # Limit cast to prevent huge responses
            
            for cp, person in cast_entries:
                cast_member = self._build_cast_member(cp, person)
                if cast_member:
                    cast_crew['cast'].append(cast_member)
            
            # Get crew members
            crew_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'crew'
            ).all()
            
            # Categorize crew members
            for cp, person in crew_entries:
                crew_member = self._build_crew_member(cp, person)
                if crew_member:
                    category = self._categorize_crew_member(cp)
                    if category in cast_crew['crew']:
                        cast_crew['crew'][category].append(crew_member)
            
            # Sort crew by importance
            self._sort_crew_members(cast_crew['crew'])
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew from database: {e}")
            return cast_crew
    
    def _build_cast_member(self, content_person: Any, person: Any) -> Optional[Dict]:
        """Build cast member data structure"""
        try:
            # Ensure person has a slug
            if not person.slug:
                try:
                    person.slug = SlugManager.generate_unique_slug(
                        self.db, self.Person, person.name, content_type='person',
                        existing_id=person.id, tmdb_id=person.tmdb_id
                    )
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to generate slug for person {person.id}: {e}")
                    person.slug = f"person-{person.id}"
            
            return {
                'id': person.id,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'character': content_person.character or '',
                'profile_path': self._format_profile_url(person.profile_path),
                'slug': person.slug,
                'popularity': person.popularity or 0,
                'order': content_person.order or 999,
                'known_for_department': person.known_for_department,
                'gender': person.gender
            }
            
        except Exception as e:
            logger.error(f"Error building cast member: {e}")
            return None
    
    def _build_crew_member(self, content_person: Any, person: Any) -> Optional[Dict]:
        """Build crew member data structure"""
        try:
            # Ensure person has a slug
            if not person.slug:
                try:
                    person.slug = SlugManager.generate_unique_slug(
                        self.db, self.Person, person.name, content_type='person',
                        existing_id=person.id, tmdb_id=person.tmdb_id
                    )
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to generate slug for person {person.id}: {e}")
                    person.slug = f"person-{person.id}"
            
            return {
                'id': person.id,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'job': content_person.job or '',
                'department': content_person.department or '',
                'profile_path': self._format_profile_url(person.profile_path),
                'slug': person.slug,
                'popularity': person.popularity or 0,
                'known_for_department': person.known_for_department
            }
            
        except Exception as e:
            logger.error(f"Error building crew member: {e}")
            return None
    
    def _categorize_crew_member(self, content_person: Any) -> str:
        """Categorize crew member into appropriate department"""
        try:
            job = (content_person.job or '').lower()
            department = (content_person.department or '').lower()
            
            # Directors
            if department == 'directing' or 'director' in job:
                return 'directors'
            
            # Writers
            elif department == 'writing' or any(keyword in job for keyword in [
                'writer', 'screenplay', 'story', 'novel', 'characters', 'script'
            ]):
                return 'writers'
            
            # Producers
            elif department == 'production' or 'producer' in job:
                return 'producers'
            
            # Other crew
            else:
                return 'other_crew'
                
        except Exception as e:
            logger.error(f"Error categorizing crew member: {e}")
            return 'other_crew'
    
    def _sort_crew_members(self, crew_dict: Dict[str, List[Dict]]):
        """Sort crew members by importance/popularity"""
        try:
            for category, members in crew_dict.items():
                # Sort by popularity, then by name
                members.sort(key=lambda x: (-x.get('popularity', 0), x.get('name', '')))
                
                # Limit each category to prevent huge responses
                if category == 'directors':
                    crew_dict[category] = members[:5]
                elif category == 'writers':
                    crew_dict[category] = members[:8]
                elif category == 'producers':
                    crew_dict[category] = members[:10]
                else:
                    crew_dict[category] = members[:15]
                    
        except Exception as e:
            logger.error(f"Error sorting crew members: {e}")
    
    @handle_api_error
    def _fetch_and_save_all_cast_crew(self, content) -> Dict:
        """Fetch cast and crew from external API and save to database"""
        cast_crew = self._empty_cast_crew()
        
        try:
            from .tmdb_service import TMDBService
            tmdb_service = TMDBService()
            
            # Get detailed content data with credits
            tmdb_data = tmdb_service.get_content_details(content.tmdb_id, content.content_type)
            
            if not tmdb_data or 'credits' not in tmdb_data:
                logger.warning(f"No credits data from TMDB for content {content.id}")
                return cast_crew
            
            credits = tmdb_data['credits']
            
            # Process cast
            cast_data = credits.get('cast', [])
            logger.info(f"Processing {len(cast_data)} cast members for {content.title}")
            
            for i, cast_member in enumerate(cast_data[:25]):  # Limit to top 25 cast
                try:
                    person = self._get_or_create_person(cast_member)
                    if person:
                        content_person = self._get_or_create_content_person(
                            content.id, person.id, 'cast',
                            character=cast_member.get('character'),
                            order=i
                        )
                        
                        cast_member_data = self._build_cast_member_from_api(cast_member, person, i)
                        if cast_member_data:
                            cast_crew['cast'].append(cast_member_data)
                            
                except Exception as e:
                    logger.warning(f"Error processing cast member: {e}")
                    continue
            
            # Process crew
            crew_data = credits.get('crew', [])
            logger.info(f"Processing {len(crew_data)} crew members for {content.title}")
            
            crew_categories = {'directors': [], 'writers': [], 'producers': [], 'other_crew': []}
            
            for crew_member in crew_data:
                try:
                    person = self._get_or_create_person(crew_member)
                    if person:
                        content_person = self._get_or_create_content_person(
                            content.id, person.id, 'crew',
                            job=crew_member.get('job'),
                            department=crew_member.get('department')
                        )
                        
                        crew_member_data = self._build_crew_member_from_api(crew_member, person)
                        if crew_member_data:
                            category = self._categorize_crew_member_from_api(crew_member)
                            crew_categories[category].append(crew_member_data)
                            
                except Exception as e:
                    logger.warning(f"Error processing crew member: {e}")
                    continue
            
            # Sort and limit crew categories
            self._sort_crew_members(crew_categories)
            cast_crew['crew'] = crew_categories
            
            # Commit all changes
            self.db.session.commit()
            
            total_saved = len(cast_crew['cast']) + sum(len(crew_list) for crew_list in cast_crew['crew'].values())
            logger.info(f"Successfully saved {total_saved} cast/crew members for content {content.id}")
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error fetching cast/crew from external API: {e}")
            self.db.session.rollback()
            return cast_crew
    
    def _get_or_create_person(self, person_data: Dict) -> Optional[Any]:
        """Get or create person from API data"""
        try:
            tmdb_id = person_data.get('id')
            if not tmdb_id:
                return None
            
            # Check if person already exists
            person = self.Person.query.filter_by(tmdb_id=tmdb_id).first()
            
            if not person:
                # Create new person
                name = person_data.get('name', 'Unknown')
                slug = SlugManager.generate_unique_slug(
                    self.db, self.Person, name, content_type='person', tmdb_id=tmdb_id
                )
                
                person = self.Person(
                    slug=slug,
                    tmdb_id=tmdb_id,
                    name=name,
                    profile_path=person_data.get('profile_path'),
                    popularity=person_data.get('popularity', 0),
                    known_for_department=person_data.get('known_for_department'),
                    gender=person_data.get('gender')
                )
                
                self.db.session.add(person)
                self.db.session.flush()
                
                logger.debug(f"Created new person: {name}")
            else:
                # Update existing person if needed
                self._update_person_if_needed(person, person_data)
            
            return person
            
        except Exception as e:
            logger.error(f"Error creating/getting person: {e}")
            return None
    
    def _update_person_if_needed(self, person: Any, person_data: Dict):
        """Update person data if needed"""
        try:
            updated = False
            
            # Update slug if missing or default
            if not person.slug or person.slug.startswith('person-'):
                person.slug = SlugManager.generate_unique_slug(
                    self.db, self.Person, person.name, content_type='person',
                    existing_id=person.id, tmdb_id=person.tmdb_id
                )
                updated = True
            
            # Update popularity if higher
            new_popularity = person_data.get('popularity', 0)
            if new_popularity > (person.popularity or 0):
                person.popularity = new_popularity
                updated = True
            
            # Update known_for_department if missing
            if not person.known_for_department and person_data.get('known_for_department'):
                person.known_for_department = person_data['known_for_department']
                updated = True
            
            if updated:
                logger.debug(f"Updated person: {person.name}")
                
        except Exception as e:
            logger.error(f"Error updating person: {e}")
    
    def _get_or_create_content_person(self, content_id: int, person_id: int, role_type: str,
                                     character: Optional[str] = None, job: Optional[str] = None,
                                     department: Optional[str] = None, order: Optional[int] = None) -> Optional[Any]:
        """Get or create content-person relationship"""
        try:
            # Check if relationship already exists
            existing = None
            
            if role_type == 'cast' and character:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character
                ).first()
            elif role_type == 'crew' and job:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    job=job
                ).first()
            
            if not existing:
                content_person = self.ContentPerson(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character,
                    job=job,
                    department=department,
                    order=order
                )
                
                self.db.session.add(content_person)
                return content_person
            
            return existing
            
        except Exception as e:
            logger.error(f"Error creating content-person relationship: {e}")
            return None
    
    def _build_cast_member_from_api(self, cast_data: Dict, person: Any, order: int) -> Optional[Dict]:
        """Build cast member from API data"""
        try:
            return {
                'id': person.id,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'character': cast_data.get('character', ''),
                'profile_path': self._format_profile_url(person.profile_path),
                'slug': person.slug,
                'popularity': person.popularity or 0,
                'order': order,
                'known_for_department': person.known_for_department,
                'gender': person.gender
            }
        except Exception as e:
            logger.error(f"Error building cast member from API: {e}")
            return None
    
    def _build_crew_member_from_api(self, crew_data: Dict, person: Any) -> Optional[Dict]:
        """Build crew member from API data"""
        try:
            return {
                'id': person.id,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'job': crew_data.get('job', ''),
                'department': crew_data.get('department', ''),
                'profile_path': self._format_profile_url(person.profile_path),
                'slug': person.slug,
                'popularity': person.popularity or 0,
                'known_for_department': person.known_for_department
            }
        except Exception as e:
            logger.error(f"Error building crew member from API: {e}")
            return None
    
    def _categorize_crew_member_from_api(self, crew_data: Dict) -> str:
        """Categorize crew member from API data"""
        try:
            job = (crew_data.get('job', '') or '').lower()
            department = (crew_data.get('department', '') or '').lower()
            
            if department == 'directing' or 'director' in job:
                return 'directors'
            elif department == 'writing' or any(keyword in job for keyword in [
                'writer', 'screenplay', 'story', 'novel', 'characters', 'script'
            ]):
                return 'writers'
            elif department == 'production' or 'producer' in job:
                return 'producers'
            else:
                return 'other_crew'
                
        except Exception:
            return 'other_crew'
    
    def _format_profile_url(self, profile_path: Optional[str]) -> Optional[str]:
        """Format profile image URL"""
        if not profile_path:
            return None
        
        if profile_path.startswith('http'):
            return profile_path
        
        return f"https://image.tmdb.org/t/p/w185{profile_path}"
    
    def _empty_cast_crew(self) -> Dict:
        """Return empty cast/crew structure"""
        return {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': [],
                'other_crew': []
            }
        }
    
    def get_person_filmography(self, person_id: int) -> Dict:
        """Get comprehensive filmography for a person"""
        try:
            filmography = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'as_other_crew': [],
                'statistics': {}
            }
            
            # Get all content relationships for this person
            filmography_entries = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person_id
            ).order_by(
                self.Content.release_date.desc().nullslast()
            ).all()
            
            for cp, content in filmography_entries:
                work_item = self._build_filmography_item(cp, content)
                if work_item:
                    if cp.role_type == 'cast':
                        filmography['as_actor'].append(work_item)
                    elif cp.role_type == 'crew':
                        category = self._get_filmography_category(cp)
                        filmography[category].append(work_item)
            
            # Calculate statistics
            filmography['statistics'] = self._calculate_filmography_stats(filmography_entries)
            
            return filmography
            
        except Exception as e:
            logger.error(f"Error getting person filmography: {e}")
            return {'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [], 'as_other_crew': [], 'statistics': {}}
    
    def _build_filmography_item(self, content_person: Any, content: Any) -> Optional[Dict]:
        """Build filmography item"""
        try:
            return {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'year': content.release_date.year if content.release_date else None,
                'rating': content.rating,
                'poster_path': self._format_profile_url(content.poster_path),
                'character': content_person.character,
                'job': content_person.job,
                'department': content_person.department
            }
        except Exception as e:
            logger.error(f"Error building filmography item: {e}")
            return None
    
    def _get_filmography_category(self, content_person: Any) -> str:
        """Get filmography category for crew member"""
        job = (content_person.job or '').lower()
        department = (content_person.department or '').lower()
        
        if department == 'directing' or 'director' in job:
            return 'as_director'
        elif department == 'writing' or any(keyword in job for keyword in ['writer', 'screenplay', 'story']):
            return 'as_writer'
        elif department == 'production' or 'producer' in job:
            return 'as_producer'
        else:
            return 'as_other_crew'
    
    def _calculate_filmography_stats(self, filmography_entries: List[Tuple]) -> Dict:
        """Calculate filmography statistics"""
        try:
            stats = {
                'total_projects': len(filmography_entries),
                'years_active': 0,
                'debut_year': None,
                'latest_year': None,
                'average_rating': 0,
                'highest_rated_work': None
            }
            
            years = []
            ratings = []
            
            for cp, content in filmography_entries:
                if content.release_date:
                    years.append(content.release_date.year)
                
                if content.rating and content.rating > 0:
                    ratings.append(content.rating)
            
            if years:
                stats['debut_year'] = min(years)
                stats['latest_year'] = max(years)
                stats['years_active'] = max(years) - min(years) + 1
            
            if ratings:
                stats['average_rating'] = round(sum(ratings) / len(ratings), 1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating filmography stats: {e}")
            return {}