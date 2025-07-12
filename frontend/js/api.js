// api.js - API communication module for Movie Recommendation App

class APIClient {
    constructor() {
        // Use environment variable or fallback to localhost
        this.baseURL = process.env.API_BASE_URL || 'http://127.0.0.1:5000';
        this.token = localStorage.getItem('authToken');
        this.refreshToken = localStorage.getItem('refreshToken');
    }

    // Set authorization token
    setToken(token) {
        this.token = token;
        localStorage.setItem('authToken', token);
    }

    // Remove authorization token
    removeToken() {
        this.token = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
    }

    // Get authorization headers
    getAuthHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        return headers;
    }

    // Generic API request method
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: this.getAuthHeaders(),
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            // Handle different response types
            const contentType = response.headers.get('content-type');
            let data;
            
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }

            if (!response.ok) {
                // Handle token expiration
                if (response.status === 401 && this.token) {
                    this.removeToken();
                    window.location.href = '/login.html';
                    return;
                }
                
                throw new Error(data.error || data.message || `HTTP ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    // Authentication endpoints
    async register(userData) {
        return this.request('/api/register', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
    }

    async login(credentials) {
        const response = await this.request('/api/login', {
            method: 'POST',
            body: JSON.stringify(credentials)
        });
        
        if (response.token) {
            this.setToken(response.token);
        }
        
        return response;
    }

    async logout() {
        this.removeToken();
        return { success: true };
    }

    // Test endpoints
    async testAPI() {
        return this.request('/api/test');
    }

    async testLogin(credentials) {
        return this.request('/api/test-login', {
            method: 'POST',
            body: JSON.stringify(credentials)
        });
    }

    // Content endpoints
    async getHomepage() {
        return this.request('/api/homepage');
    }

    async getRecommendations() {
        return this.request('/api/recommendations');
    }

    async getContentDetails(contentId) {
        return this.request(`/api/content/${contentId}`);
    }

    async getTMDBContent(tmdbId) {
        return this.request(`/api/content/tmdb/${tmdbId}`);
    }

    async getWatchOptions(contentId) {
        return this.request(`/api/content/${contentId}/watch-options`);
    }

    // Search endpoints
    async searchContent(query, type = 'movie') {
        const params = new URLSearchParams({ q: query, type });
        return this.request(`/api/search?${params}`);
    }

    // User interaction endpoints
    async recordInteraction(interactionData) {
        return this.request('/api/interact', {
            method: 'POST',
            body: JSON.stringify(interactionData)
        });
    }

    // Convenience methods for common interactions
    async likeContent(contentId) {
        return this.recordInteraction({
            content_id: contentId,
            interaction_type: 'like'
        });
    }

    async favoriteContent(contentId) {
        return this.recordInteraction({
            content_id: contentId,
            interaction_type: 'favorite'
        });
    }

    async addToWishlist(contentId) {
        return this.recordInteraction({
            content_id: contentId,
            interaction_type: 'wishlist'
        });
    }

    async rateContent(contentId, rating) {
        return this.recordInteraction({
            content_id: contentId,
            interaction_type: 'view',
            rating: rating
        });
    }

    // Admin endpoints
    async adminCurate(curationData) {
        return this.request('/api/admin/curate', {
            method: 'POST',
            body: JSON.stringify(curationData)
        });
    }

    async getAdminDashboard() {
        return this.request('/api/admin/dashboard');
    }

    // Content sync
    async syncContent() {
        return this.request('/api/sync-content', {
            method: 'POST'
        });
    }

    // Health check
    async healthCheck() {
        return this.request('/health');
    }

    // Utility methods
    isAuthenticated() {
        return !!this.token;
    }

    getUserId() {
        if (!this.token) return null;
        
        try {
            const payload = JSON.parse(atob(this.token.split('.')[1]));
            return payload.sub || payload.user_id;
        } catch (error) {
            console.error('Error parsing token:', error);
            return null;
        }
    }

    // Image URL helpers
    getTMDBImageURL(path, size = 'w500') {
        if (!path) return null;
        return `https://image.tmdb.org/t/p/${size}${path}`;
    }

    getBackdropURL(path, size = 'w1280') {
        if (!path) return null;
        return `https://image.tmdb.org/t/p/${size}${path}`;
    }

    // Error handling helper
    handleError(error) {
        console.error('API Error:', error);
        
        if (error.message.includes('401')) {
            this.removeToken();
            window.location.href = '/login.html';
        }
        
        return {
            error: true,
            message: error.message || 'An unexpected error occurred'
        };
    }

    // Batch requests for better performance
    async batchRequest(requests) {
        const promises = requests.map(({ endpoint, options }) => 
            this.request(endpoint, options).catch(error => ({ error: error.message }))
        );
        
        return Promise.all(promises);
    }

    // Cache management
    clearCache() {
        // Clear any cached data if implemented
        localStorage.removeItem('contentCache');
        localStorage.removeItem('userPreferences');
    }

    // Network status check
    async checkConnection() {
        try {
            await this.healthCheck();
            return true;
        } catch (error) {
            return false;
        }
    }
}

// Create singleton instance
const apiClient = new APIClient();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = apiClient;
} else {
    window.apiClient = apiClient;
}

// Additional utility functions
const APIUtils = {
    // Format content for display
    formatContent(content) {
        return {
            ...content,
            posterURL: apiClient.getTMDBImageURL(content.poster_path),
            backdropURL: apiClient.getBackdropURL(content.backdrop_path),
            releaseYear: content.release_date ? new Date(content.release_date).getFullYear() : null,
            formattedRating: content.rating ? Math.round(content.rating * 10) / 10 : null
        };
    },

    // Format streaming platforms
    formatStreamingPlatforms(platforms) {
        return {
            free: platforms.free || [],
            paid: platforms.paid || [],
            all: platforms.all || []
        };
    },

    // Handle API errors gracefully
    async safeAPICall(apiCall, fallback = null) {
        try {
            return await apiCall();
        } catch (error) {
            console.error('Safe API call failed:', error);
            return fallback;
        }
    },

    // Debounced search function
    debounce(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }
};

// Export utilities
if (typeof module !== 'undefined' && module.exports) {
    module.exports.APIUtils = APIUtils;
} else {
    window.APIUtils = APIUtils;
}