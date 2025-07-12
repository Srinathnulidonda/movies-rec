// app.js - Main application logic for Movie Recommendation App
class MovieApp {
    constructor() {
        this.apiUrl = 'https://backend-app-970m.onrender.com/api';
        this.currentUser = null;
        this.authToken = null;
        this.currentPage = 'home';
        this.searchResults = [];
        this.recommendations = [];
        this.watchlist = [];
        this.favorites = [];
        
        this.init();
    }

    // Initialize the application
    init() {
        this.loadStoredAuth();
        this.setupEventListeners();
        this.setupRouter();
        this.loadInitialData();
    }

    // Load stored authentication data
    loadStoredAuth() {
        const token = localStorage.getItem('authToken');
        const user = localStorage.getItem('currentUser');
        
        if (token && user) {
            this.authToken = token;
            this.currentUser = JSON.parse(user);
            this.setupAuthenticatedState();
        }
    }

    // Setup event listeners for the application
    setupEventListeners() {
        // Navigation
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-nav]')) {
                e.preventDefault();
                this.navigateTo(e.target.dataset.nav);
            }
        });

        // Search functionality
        const searchInput = document.getElementById('searchInput');
        const searchButton = document.getElementById('searchButton');
        
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce(this.handleSearch.bind(this), 300));
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch();
                }
            });
        }

        if (searchButton) {
            searchButton.addEventListener('click', this.handleSearch.bind(this));
        }

        // Authentication forms
        this.setupAuthForms();

        // Content interaction handlers
        this.setupContentInteractionHandlers();

        // Modal handlers
        this.setupModalHandlers();

        // Responsive navigation
        this.setupMobileNavigation();
    }

    // Setup authentication forms
    setupAuthForms() {
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');

        if (loginForm) {
            loginForm.addEventListener('submit', this.handleLogin.bind(this));
        }

        if (registerForm) {
            registerForm.addEventListener('submit', this.handleRegister.bind(this));
        }

        // Toggle between login and register
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-toggle-auth]')) {
                this.toggleAuthForm();
            }
        });
    }

    // Setup content interaction handlers
    setupContentInteractionHandlers() {
        document.addEventListener('click', (e) => {
            const target = e.target.closest('[data-action]');
            if (!target) return;

            const action = target.dataset.action;
            const contentId = target.dataset.contentId;

            switch (action) {
                case 'like':
                    this.handleLike(contentId);
                    break;
                case 'favorite':
                    this.handleFavorite(contentId);
                    break;
                case 'watchlist':
                    this.handleWatchlist(contentId);
                    break;
                case 'view-details':
                    this.viewContentDetails(contentId);
                    break;
                case 'watch-options':
                    this.showWatchOptions(contentId);
                    break;
                case 'rate':
                    this.showRatingModal(contentId);
                    break;
            }
        });
    }

    // Setup modal handlers
    setupModalHandlers() {
        document.addEventListener('click', (e) => {
            if (e.target.matches('.modal-close') || e.target.matches('.modal-overlay')) {
                this.closeModal();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    // Setup mobile navigation
    setupMobileNavigation() {
        const mobileMenuButton = document.getElementById('mobileMenuButton');
        const mobileMenu = document.getElementById('mobileMenu');

        if (mobileMenuButton && mobileMenu) {
            mobileMenuButton.addEventListener('click', () => {
                mobileMenu.classList.toggle('active');
            });
        }
    }

    // Simple router implementation
    setupRouter() {
        window.addEventListener('popstate', () => {
            this.handleRouteChange();
        });
        
        this.handleRouteChange();
    }

    // Handle route changes
    handleRouteChange() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'home';
        this.navigateTo(page, false);
    }

    // Navigate to a specific page
    navigateTo(page, pushState = true) {
        if (pushState) {
            history.pushState(null, '', `/${page}`);
        }

        this.currentPage = page;
        this.renderPage(page);
        this.updateNavigation(page);
    }

    // Update navigation active state
    updateNavigation(currentPage) {
        document.querySelectorAll('[data-nav]').forEach(link => {
            link.classList.toggle('active', link.dataset.nav === currentPage);
        });
    }

    // Render the current page
    async renderPage(page) {
        const mainContent = document.getElementById('mainContent');
        if (!mainContent) return;

        this.showLoader();

        try {
            switch (page) {
                case 'home':
                    await this.renderHomePage();
                    break;
                case 'search':
                    await this.renderSearchPage();
                    break;
                case 'recommendations':
                    await this.renderRecommendationsPage();
                    break;
                case 'watchlist':
                    await this.renderWatchlistPage();
                    break;
                case 'favorites':
                    await this.renderFavoritesPage();
                    break;
                case 'profile':
                    await this.renderProfilePage();
                    break;
                case 'login':
                    this.renderLoginPage();
                    break;
                default:
                    await this.renderHomePage();
            }
        } catch (error) {
            console.error('Error rendering page:', error);
            this.showError('Failed to load page content');
        } finally {
            this.hideLoader();
        }
    }

    // Load initial data
    async loadInitialData() {
        try {
            if (this.currentUser) {
                await this.loadUserData();
            }
            await this.loadHomepageData();
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    // Load user-specific data
    async loadUserData() {
        try {
            const [recommendations, watchlist, favorites] = await Promise.all([
                this.apiRequest('/recommendations'),
                this.apiRequest('/user/watchlist'),
                this.apiRequest('/user/favorites')
            ]);

            this.recommendations = recommendations.hybrid_recommendations || [];
            this.watchlist = watchlist || [];
            this.favorites = favorites || [];
        } catch (error) {
            console.error('Error loading user data:', error);
        }
    }

    // Load homepage data
    async loadHomepageData() {
        try {
            const homepageData = await this.apiRequest('/homepage');
            this.homepageData = homepageData;
        } catch (error) {
            console.error('Error loading homepage data:', error);
            this.homepageData = null;
        }
    }

    // Authentication handlers
    async handleLogin(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const credentials = {
            username: formData.get('username'),
            password: formData.get('password')
        };

        try {
            this.showLoader();
            const response = await this.apiRequest('/login', 'POST', credentials);
            
            this.authToken = response.token;
            this.currentUser = {
                id: response.user_id,
                username: response.username
            };

            // Store auth data
            localStorage.setItem('authToken', this.authToken);
            localStorage.setItem('currentUser', JSON.stringify(this.currentUser));

            this.setupAuthenticatedState();
            this.navigateTo('home');
            this.showSuccess('Login successful!');
            
        } catch (error) {
            this.showError(error.message || 'Login failed');
        } finally {
            this.hideLoader();
        }
    }

    async handleRegister(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const userData = {
            username: formData.get('username'),
            email: formData.get('email'),
            password: formData.get('password')
        };

        // Basic validation
        if (userData.password !== formData.get('confirmPassword')) {
            this.showError('Passwords do not match');
            return;
        }

        try {
            this.showLoader();
            const response = await this.apiRequest('/register', 'POST', userData);
            
            this.authToken = response.token;
            this.currentUser = {
                id: response.user_id,
                username: response.username
            };

            // Store auth data
            localStorage.setItem('authToken', this.authToken);
            localStorage.setItem('currentUser', JSON.stringify(this.currentUser));

            this.setupAuthenticatedState();
            this.navigateTo('home');
            this.showSuccess('Registration successful!');
            
        } catch (error) {
            this.showError(error.message || 'Registration failed');
        } finally {
            this.hideLoader();
        }
    }

    // Setup authenticated state
    setupAuthenticatedState() {
        document.body.classList.add('authenticated');
        this.updateUserDisplay();
        this.loadUserData();
    }

    // Update user display in UI
    updateUserDisplay() {
        const userDisplays = document.querySelectorAll('.user-display');
        userDisplays.forEach(display => {
            display.textContent = this.currentUser.username;
        });

        const authButtons = document.querySelectorAll('.auth-required');
        authButtons.forEach(button => {
            button.style.display = this.currentUser ? 'block' : 'none';
        });
    }

    // Logout functionality
    logout() {
        this.authToken = null;
        this.currentUser = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('currentUser');
        
        document.body.classList.remove('authenticated');
        this.navigateTo('home');
        this.showSuccess('Logged out successfully');
    }

    // Search functionality
    async handleSearch() {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput?.value.trim();
        
        if (!query) return;

        try {
            this.showLoader();
            const results = await this.apiRequest(`/search?q=${encodeURIComponent(query)}`);
            this.searchResults = [
                ...(results.database_results || []),
                ...(results.tmdb_results || [])
            ];
            
            if (this.currentPage !== 'search') {
                this.navigateTo('search');
            } else {
                this.renderSearchResults();
            }
        } catch (error) {
            this.showError('Search failed');
        } finally {
            this.hideLoader();
        }
    }

    // Content interaction handlers
    async handleLike(contentId) {
        if (!this.currentUser) {
            this.showError('Please login to like content');
            return;
        }

        try {
            await this.apiRequest('/interact', 'POST', {
                content_id: contentId,
                interaction_type: 'like'
            });
            this.showSuccess('Content liked!');
            this.updateContentInteractionUI(contentId, 'like');
        } catch (error) {
            this.showError('Failed to like content');
        }
    }

    async handleFavorite(contentId) {
        if (!this.currentUser) {
            this.showError('Please login to add favorites');
            return;
        }

        try {
            await this.apiRequest('/interact', 'POST', {
                content_id: contentId,
                interaction_type: 'favorite'
            });
            this.showSuccess('Added to favorites!');
            this.updateContentInteractionUI(contentId, 'favorite');
        } catch (error) {
            this.showError('Failed to add to favorites');
        }
    }

    async handleWatchlist(contentId) {
        if (!this.currentUser) {
            this.showError('Please login to add to watchlist');
            return;
        }

        try {
            await this.apiRequest('/interact', 'POST', {
                content_id: contentId,
                interaction_type: 'watchlist'
            });
            this.showSuccess('Added to watchlist!');
            this.updateContentInteractionUI(contentId, 'watchlist');
        } catch (error) {
            this.showError('Failed to add to watchlist');
        }
    }

    // Update UI after content interaction
    updateContentInteractionUI(contentId, interactionType) {
        const buttons = document.querySelectorAll(`[data-content-id="${contentId}"][data-action="${interactionType}"]`);
        buttons.forEach(button => {
            button.classList.add('active');
        });
    }

    // Show content details
    async viewContentDetails(contentId) {
        try {
            this.showLoader();
            const contentData = await this.apiRequest(`/content/${contentId}`);
            this.showContentModal(contentData);
        } catch (error) {
            this.showError('Failed to load content details');
        } finally {
            this.hideLoader();
        }
    }

    // Show watch options
    async showWatchOptions(contentId) {
        try {
            this.showLoader();
            const watchOptions = await this.apiRequest(`/content/${contentId}/watch-options`);
            this.showWatchOptionsModal(watchOptions);
        } catch (error) {
            this.showError('Failed to load watch options');
        } finally {
            this.hideLoader();
        }
    }

    // Show rating modal
    showRatingModal(contentId) {
        const modal = document.getElementById('ratingModal');
        if (modal) {
            modal.dataset.contentId = contentId;
            modal.style.display = 'block';
        }
    }

    // Page rendering methods
    async renderHomePage() {
        const mainContent = document.getElementById('mainContent');
        const data = this.homepageData;
        
        if (!data) {
            mainContent.innerHTML = '<div class="error">Failed to load homepage data</div>';
            return;
        }

        mainContent.innerHTML = `
            <div class="homepage">
                <section class="hero-section">
                    <div class="hero-content">
                        <h1>Discover Your Next Favorite Movie</h1>
                        <p>Personalized recommendations just for you</p>
                        <div class="hero-search">
                            <input type="text" id="heroSearch" placeholder="Search movies, TV shows, anime...">
                            <button onclick="app.handleHeroSearch()">Search</button>
                        </div>
                    </div>
                </section>

                <section class="content-sections">
                    ${this.renderContentSection('Trending Movies', data.trending.movies)}
                    ${this.renderContentSection('Trending TV Shows', data.trending.tv)}
                    ${this.renderContentSection('Popular Anime', data.trending.anime)}
                    ${this.renderGenreSection('Popular by Genre', data.popular_by_genre)}
                    ${this.renderContentSection('Critics Choice', data.critics_choice)}
                </section>
            </div>
        `;
    }

    async renderRecommendationsPage() {
        if (!this.currentUser) {
            this.navigateTo('login');
            return;
        }

        const mainContent = document.getElementById('mainContent');
        
        mainContent.innerHTML = `
            <div class="recommendations-page">
                <h1>Your Recommendations</h1>
                ${this.renderContentSection('Recommended for You', this.recommendations)}
                ${this.renderContentSection('Based on Your Favorites', this.recommendations)}
            </div>
        `;
    }

    renderSearchPage() {
        const mainContent = document.getElementById('mainContent');
        
        mainContent.innerHTML = `
            <div class="search-page">
                <h1>Search Results</h1>
                <div class="search-results">
                    ${this.renderSearchResults()}
                </div>
            </div>
        `;
    }

    renderSearchResults() {
        if (!this.searchResults.length) {
            return '<div class="no-results">No results found</div>';
        }

        return this.searchResults.map(item => this.renderContentCard(item)).join('');
    }

    // Utility rendering methods
    renderContentSection(title, items) {
        if (!items || !items.length) return '';

        return `
            <div class="content-section">
                <h2>${title}</h2>
                <div class="content-grid">
                    ${items.map(item => this.renderContentCard(item)).join('')}
                </div>
            </div>
        `;
    }

    renderGenreSection(title, genreData) {
        if (!genreData) return '';

        return `
            <div class="genre-section">
                <h2>${title}</h2>
                ${Object.entries(genreData).map(([genre, items]) => 
                    this.renderContentSection(genre, items)
                ).join('')}
            </div>
        `;
    }

    renderContentCard(item) {
        const posterUrl = item.poster_path 
            ? `https://image.tmdb.org/t/p/w500${item.poster_path}`
            : 'https://via.placeholder.com/300x450?text=No+Image';

        return `
            <div class="content-card" data-content-id="${item.id || item.tmdb_id}">
                <div class="content-image">
                    <img src="${posterUrl}" alt="${item.title || item.name}" loading="lazy">
                    <div class="content-overlay">
                        <button class="btn-primary" data-action="view-details" data-content-id="${item.id || item.tmdb_id}">
                            View Details
                        </button>
                    </div>
                </div>
                <div class="content-info">
                    <h3>${item.title || item.name}</h3>
                    <p class="content-year">${this.extractYear(item.release_date || item.first_air_date)}</p>
                    <div class="content-rating">
                        <span class="rating-star">★</span>
                        <span>${(item.vote_average || item.rating || 0).toFixed(1)}</span>
                    </div>
                    <div class="content-actions">
                        <button class="action-btn" data-action="favorite" data-content-id="${item.id || item.tmdb_id}">
                            <i class="icon-heart"></i>
                        </button>
                        <button class="action-btn" data-action="watchlist" data-content-id="${item.id || item.tmdb_id}">
                            <i class="icon-bookmark"></i>
                        </button>
                        <button class="action-btn" data-action="watch-options" data-content-id="${item.id || item.tmdb_id}">
                            <i class="icon-play"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    // Modal methods
    showContentModal(contentData) {
        const modal = document.getElementById('contentModal') || this.createContentModal();
        const content = contentData.content;
        
        modal.innerHTML = `
            <div class="modal-overlay" onclick="app.closeModal()">
                <div class="modal-content content-modal" onclick="event.stopPropagation()">
                    <button class="modal-close" onclick="app.closeModal()">&times;</button>
                    <div class="modal-header">
                        <img src="https://image.tmdb.org/t/p/w500${content.poster_path}" alt="${content.title}">
                        <div class="modal-info">
                            <h2>${content.title}</h2>
                            <p class="modal-overview">${content.overview}</p>
                            <div class="modal-meta">
                                <span class="rating">★ ${content.rating}/10</span>
                                <span class="runtime">${content.runtime} min</span>
                                <span class="year">${this.extractYear(content.release_date)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="modal-actions">
                        <button class="btn-primary" data-action="watch-options" data-content-id="${content.id}">
                            Watch Options
                        </button>
                        <button class="btn-secondary" data-action="favorite" data-content-id="${content.id}">
                            Add to Favorites
                        </button>
                        <button class="btn-secondary" data-action="watchlist" data-content-id="${content.id}">
                            Add to Watchlist
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    showWatchOptionsModal(watchOptions) {
        const modal = document.getElementById('watchOptionsModal') || this.createWatchOptionsModal();
        
        modal.innerHTML = `
            <div class="modal-overlay" onclick="app.closeModal()">
                <div class="modal-content watch-options-modal" onclick="event.stopPropagation()">
                    <button class="modal-close" onclick="app.closeModal()">&times;</button>
                    <h2>Watch Options for ${watchOptions.title}</h2>
                    
                    ${watchOptions.streaming_platforms.free.length > 0 ? `
                        <div class="platform-section">
                            <h3>Free Streaming</h3>
                            <div class="platform-list">
                                ${watchOptions.streaming_platforms.free.map(platform => `
                                    <div class="platform-item">
                                        <span class="platform-name">${platform.platform_name}</span>
                                        <a href="${platform.watch_url}" target="_blank" class="btn-primary">Watch Now</a>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${watchOptions.streaming_platforms.paid.length > 0 ? `
                        <div class="platform-section">
                            <h3>Paid Streaming</h3>
                            <div class="platform-list">
                                ${watchOptions.streaming_platforms.paid.map(platform => `
                                    <div class="platform-item">
                                        <span class="platform-name">${platform.platform_name}</span>
                                        <a href="${platform.watch_url}" target="_blank" class="btn-primary">Watch Now</a>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${watchOptions.theater_info.length > 0 ? `
                        <div class="platform-section">
                            <h3>In Theaters</h3>
                            <div class="theater-list">
                                ${watchOptions.theater_info.map(theater => `
                                    <div class="theater-item">
                                        <div class="theater-info">
                                            <h4>${theater.theater_name}</h4>
                                            <p>${theater.theater_address}</p>
                                            <p>Showtime: ${theater.showtime}</p>
                                            <p>Price: $${theater.ticket_price}</p>
                                        </div>
                                        <a href="${theater.booking_url}" target="_blank" class="btn-primary">Book Tickets</a>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    createContentModal() {
        const modal = document.createElement('div');
        modal.id = 'contentModal';
        modal.className = 'modal';
        document.body.appendChild(modal);
        return modal;
    }

    createWatchOptionsModal() {
        const modal = document.createElement('div');
        modal.id = 'watchOptionsModal';
        modal.className = 'modal';
        document.body.appendChild(modal);
        return modal;
    }

    closeModal() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.style.display = 'none';
        });
    }

    // Utility methods
    async apiRequest(endpoint, method = 'GET', data = null) {
        const url = `${this.apiUrl}${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (this.authToken) {
            options.headers.Authorization = `Bearer ${this.authToken}`;
        }

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Request failed');
        }

        return response.json();
    }

    debounce(func, wait) {
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

    extractYear(dateString) {
        if (!dateString) return 'N/A';
        return new Date(dateString).getFullYear();
    }

    showLoader() {
        const loader = document.getElementById('loader');
        if (loader) {
            loader.style.display = 'block';
        }
    }

    hideLoader() {
        const loader = document.getElementById('loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Hero search handler
    handleHeroSearch() {
        const heroSearch = document.getElementById('heroSearch');
        const searchInput = document.getElementById('searchInput');
        
        if (heroSearch && searchInput) {
            searchInput.value = heroSearch.value;
            this.handleSearch();
        }
    }

    // Toggle authentication form
    toggleAuthForm() {
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');
        
        if (loginForm && registerForm) {
            loginForm.style.display = loginForm.style.display === 'none' ? 'block' : 'none';
            registerForm.style.display = registerForm.style.display === 'none' ? 'block' : 'none';
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MovieApp();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MovieApp;
}