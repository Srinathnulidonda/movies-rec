// recommendations.js - Frontend recommendation logic and API integration

class RecommendationService {
    constructor() {
        this.baseURL = 'https://backend-app-970m.onrender.com/api';
        this.cache = new Map();
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
    }

    // Get authentication token
    getAuthToken() {
        return localStorage.getItem('authToken');
    }

    // Get user ID from token or localStorage
    getUserId() {
        return localStorage.getItem('userId');
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!this.getAuthToken();
    }

    // Generic API call with error handling
    async apiCall(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add auth token if available
        if (this.getAuthToken()) {
            config.headers['Authorization'] = `Bearer ${this.getAuthToken()}`;
        }

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API call failed for ${endpoint}:`, error);
            throw error;
        }
    }

    // Cache management
    setCacheItem(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }

    getCacheItem(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        if (Date.now() - item.timestamp > this.cacheExpiry) {
            this.cache.delete(key);
            return null;
        }
        
        return item.data;
    }

    // Homepage recommendations for non-authenticated users
    async getHomepageRecommendations() {
        const cacheKey = 'homepage_recommendations';
        const cached = this.getCacheItem(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const data = await this.apiCall('/homepage');
            this.setCacheItem(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Failed to fetch homepage recommendations:', error);
            return this.getFallbackHomepageData();
        }
    }

    // Personalized recommendations for authenticated users
    async getPersonalizedRecommendations() {
        if (!this.isAuthenticated()) {
            return this.getHomepageRecommendations();
        }

        const cacheKey = `personalized_recommendations_${this.getUserId()}`;
        const cached = this.getCacheItem(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const data = await this.apiCall('/recommendations');
            this.setCacheItem(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Failed to fetch personalized recommendations:', error);
            // Fallback to homepage recommendations
            return this.getHomepageRecommendations();
        }
    }

    // Get content details with watch options
    async getContentDetails(contentId) {
        const cacheKey = `content_details_${contentId}`;
        const cached = this.getCacheItem(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const data = await this.apiCall(`/content/${contentId}`);
            this.setCacheItem(cacheKey, data);
            return data;
        } catch (error) {
            console.error(`Failed to fetch content details for ${contentId}:`, error);
            throw error;
        }
    }

    // Get TMDB content details
    async getTMDBContent(tmdbId) {
        const cacheKey = `tmdb_content_${tmdbId}`;
        const cached = this.getCacheItem(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const data = await this.apiCall(`/content/tmdb/${tmdbId}`);
            this.setCacheItem(cacheKey, data);
            return data;
        } catch (error) {
            console.error(`Failed to fetch TMDB content for ${tmdbId}:`, error);
            throw error;
        }
    }

    // Search content
    async searchContent(query, type = 'movie') {
        if (!query.trim()) {
            return { database_results: [], tmdb_results: [] };
        }

        try {
            const params = new URLSearchParams({
                q: query,
                type: type
            });
            
            const data = await this.apiCall(`/search?${params}`);
            return data;
        } catch (error) {
            console.error('Search failed:', error);
            return { database_results: [], tmdb_results: [] };
        }
    }

    // Record user interaction
    async recordInteraction(contentId, interactionType, rating = null) {
        if (!this.isAuthenticated()) {
            console.log('User not authenticated, interaction not recorded');
            return;
        }

        try {
            const data = await this.apiCall('/interact', {
                method: 'POST',
                body: JSON.stringify({
                    content_id: contentId,
                    interaction_type: interactionType,
                    rating: rating
                })
            });

            // Clear personalized recommendations cache to get updated recommendations
            const cacheKey = `personalized_recommendations_${this.getUserId()}`;
            this.cache.delete(cacheKey);

            return data;
        } catch (error) {
            console.error('Failed to record interaction:', error);
            throw error;
        }
    }

    // Get watch options for content
    async getWatchOptions(contentId) {
        const cacheKey = `watch_options_${contentId}`;
        const cached = this.getCacheItem(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const data = await this.apiCall(`/content/${contentId}/watch-options`);
            this.setCacheItem(cacheKey, data);
            return data;
        } catch (error) {
            console.error(`Failed to fetch watch options for ${contentId}:`, error);
            return {
                streaming_platforms: { free: [], paid: [], all: [] },
                theater_info: [],
                in_theaters: false
            };
        }
    }

    // Trigger content sync (admin function)
    async syncContent() {
        if (!this.isAuthenticated()) {
            throw new Error('Authentication required');
        }

        try {
            const data = await this.apiCall('/sync-content', {
                method: 'POST'
            });

            // Clear all caches after sync
            this.cache.clear();
            
            return data;
        } catch (error) {
            console.error('Failed to sync content:', error);
            throw error;
        }
    }

    // Fallback data for when API fails
    getFallbackHomepageData() {
        return {
            trending: {
                movies: [],
                tv: [],
                anime: []
            },
            popular_by_genre: {
                Action: [],
                Comedy: [],
                Drama: [],
                Horror: [],
                'Sci-Fi': [],
                Romance: []
            },
            regional: {
                Telugu: [],
                Hindi: [],
                Tamil: [],
                Kannada: []
            },
            critics_choice: [],
            user_favorites: []
        };
    }

    // Utility methods for frontend
    
    // Format content for display
    formatContent(content) {
        return {
            ...content,
            poster_url: content.poster_path ? 
                `https://image.tmdb.org/t/p/w500${content.poster_path}` : 
                '/images/no-poster.jpg',
            backdrop_url: content.backdrop_path ? 
                `https://image.tmdb.org/t/p/w1280${content.backdrop_path}` : 
                '/images/no-backdrop.jpg',
            formatted_date: content.release_date ? 
                new Date(content.release_date).getFullYear() : 
                'Unknown',
            rating_stars: content.rating ? 
                Math.round(content.rating / 2) : 
                0
        };
    }

    // Get genre names from IDs
    getGenreNames(genreIds) {
        const genreMap = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
            99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
            27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        };

        if (!genreIds || !Array.isArray(genreIds)) {
            return [];
        }

        return genreIds.map(id => genreMap[id] || 'Unknown').filter(name => name !== 'Unknown');
    }

    // Filter content by preferences
    filterByPreferences(content, preferences = {}) {
        if (!preferences || Object.keys(preferences).length === 0) {
            return content;
        }

        return content.filter(item => {
            // Filter by preferred genres
            if (preferences.favorite_genres && preferences.favorite_genres.length > 0) {
                const itemGenres = this.getGenreNames(item.genres || item.genre_ids || []);
                const hasPreferredGenre = itemGenres.some(genre => 
                    preferences.favorite_genres.includes(genre)
                );
                if (!hasPreferredGenre) return false;
            }

            // Filter by minimum rating
            if (preferences.min_rating) {
                if (!item.rating || item.rating < preferences.min_rating) {
                    return false;
                }
            }

            // Filter by language
            if (preferences.preferred_languages && preferences.preferred_languages.length > 0) {
                if (!preferences.preferred_languages.includes(item.language)) {
                    return false;
                }
            }

            return true;
        });
    }

    // Sort content by various criteria
    sortContent(content, sortBy = 'popularity') {
        const sortedContent = [...content];

        switch (sortBy) {
            case 'rating':
                return sortedContent.sort((a, b) => (b.rating || 0) - (a.rating || 0));
            case 'release_date':
                return sortedContent.sort((a, b) => {
                    const dateA = new Date(a.release_date || '1900-01-01');
                    const dateB = new Date(b.release_date || '1900-01-01');
                    return dateB - dateA;
                });
            case 'title':
                return sortedContent.sort((a, b) => 
                    (a.title || a.name || '').localeCompare(b.title || b.name || '')
                );
            case 'popularity':
            default:
                return sortedContent.sort((a, b) => (b.popularity || 0) - (a.popularity || 0));
        }
    }

    // Get user's interaction history
    async getUserInteractions() {
        if (!this.isAuthenticated()) {
            return [];
        }

        try {
            // This would need to be implemented in your backend
            const data = await this.apiCall('/user/interactions');
            return data.interactions || [];
        } catch (error) {
            console.error('Failed to fetch user interactions:', error);
            return [];
        }
    }

    // Clear all caches
    clearCache() {
        this.cache.clear();
    }

    // Get recommendation explanation (for educational purposes)
    getRecommendationExplanation(content, reason) {
        const explanations = {
            'trending': `This ${content.content_type || 'content'} is currently trending and popular among users.`,
            'genre_match': `Recommended because you like ${this.getGenreNames(content.genres || []).join(', ')} content.`,
            'similar_users': `Users with similar preferences also enjoyed this ${content.content_type || 'content'}.`,
            'content_based': `Similar to other content you've liked based on plot, genre, and style.`,
            'collaborative': `Recommended based on what users with similar tastes have watched.`,
            'critics_choice': `Handpicked by our editorial team for exceptional quality.`,
            'regional': `Popular in your region and language preference.`,
            'recent_activity': `Based on your recent viewing activity and preferences.`
        };

        return explanations[reason] || 'Recommended for you based on your viewing history.';
    }
}

// Export singleton instance
const recommendationService = new RecommendationService();

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = recommendationService;
} else if (typeof window !== 'undefined') {
    window.RecommendationService = RecommendationService;
    window.recommendationService = recommendationService;
}

// Additional utility functions for the frontend

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Create debounced search function
const debouncedSearch = debounce(async (query, callback) => {
    try {
        const results = await recommendationService.searchContent(query);
        callback(results);
    } catch (error) {
        console.error('Search error:', error);
        callback({ database_results: [], tmdb_results: [] });
    }
}, 300);

// Intersection Observer for lazy loading recommendations
function createIntersectionObserver(callback, options = {}) {
    const defaultOptions = {
        root: null,
        rootMargin: '50px',
        threshold: 0.1
    };

    return new IntersectionObserver(callback, { ...defaultOptions, ...options });
}

// Export utility functions
if (typeof window !== 'undefined') {
    window.debouncedSearch = debouncedSearch;
    window.createIntersectionObserver = createIntersectionObserver;
}