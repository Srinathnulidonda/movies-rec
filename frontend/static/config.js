// Configuration file for the MovieHub frontend
const CONFIG = {
    // Backend API URL - Update this with your deployed backend URL
    API_BASE_URL: 'http://127.0.0.1:5000',
    
    // TMDB Configuration
    TMDB_IMAGE_BASE_URL: 'https://image.tmdb.org/t/p/',
    TMDB_POSTER_SIZE: 'w500',
    TMDB_BACKDROP_SIZE: 'w1280',
    
    // Pagination
    ITEMS_PER_PAGE: 20,
    
    // Cache duration (in milliseconds)
    CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
    
    // Default placeholder images
    DEFAULT_POSTER: 'https://via.placeholder.com/500x750/374151/ffffff?text=No+Image',
    DEFAULT_BACKDROP: 'https://via.placeholder.com/1280x720/374151/ffffff?text=No+Image',
    
    // API Endpoints
    ENDPOINTS: {
        REGISTER: '/register',
        LOGIN: '/login',
        MOVIES_POPULAR: '/movies/popular',
        MOVIES_SEARCH: '/movies/search',
        ANIME_POPULAR: '/anime/popular',
        SERIES_POPULAR: '/series/popular',
        WISHLIST: '/wishlist',
        FAVORITES: '/favorites',
        RECOMMENDATIONS: '/recommendations',
        WATCH_HISTORY: '/watch-history',
        ADMIN_POST: '/admin/post-suggestion'
    },
    
    // Local Storage Keys
    STORAGE_KEYS: {
        TOKEN: 'moviehub_token',
        USER_ID: 'moviehub_user_id',
        CACHE_PREFIX: 'moviehub_cache_'
    },
    
    // UI Configuration
    UI: {
        TOAST_DURATION: 3000,
        LOADING_DELAY: 500,
        ANIMATION_DURATION: 300
    },
    
    // Error Messages
    ERRORS: {
        NETWORK: 'Network error. Please check your connection.',
        UNAUTHORIZED: 'Please login to continue.',
        SERVER: 'Server error. Please try again later.',
        NOT_FOUND: 'Content not found.',
        VALIDATION: 'Please check your input.'
    }
};

// Utility function to get full image URL
function getImageUrl(path, size = CONFIG.TMDB_POSTER_SIZE) {
    if (!path) return CONFIG.DEFAULT_POSTER;
    return `${CONFIG.TMDB_IMAGE_BASE_URL}${size}${path}`;
}

// Utility function to get API URL
function getApiUrl(endpoint) {
    return `${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS[endpoint] || endpoint}`;
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, getImageUrl, getApiUrl };
}