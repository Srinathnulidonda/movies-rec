// utils.js - Utility functions for Movie Recommendation App

// API Configuration
const API_CONFIG = {
    BASE_URL: process.env.NODE_ENV === 'production' 
        ? 'https://backend-app-970m.onrender.com' 
        : 'http://localhost:5000',
    TMDB_IMAGE_BASE: 'https://image.tmdb.org/t/p/',
    TMDB_IMAGE_SIZES: {
        poster: 'w500',
        backdrop: 'w1280',
        profile: 'w185'
    }
};

// Local Storage Keys
const STORAGE_KEYS = {
    TOKEN: 'movie_app_token',
    USER: 'movie_app_user',
    FAVORITES: 'movie_app_favorites',
    WATCHLIST: 'movie_app_watchlist',
    PREFERENCES: 'movie_app_preferences'
};

// Authentication Utilities
export const auth = {
    // Get token from localStorage
    getToken() {
        return localStorage.getItem(STORAGE_KEYS.TOKEN);
    },

    // Set token in localStorage
    setToken(token) {
        localStorage.setItem(STORAGE_KEYS.TOKEN, token);
    },

    // Remove token from localStorage
    removeToken() {
        localStorage.removeItem(STORAGE_KEYS.TOKEN);
    },

    // Check if user is authenticated
    isAuthenticated() {
        return !!this.getToken();
    },

    // Get current user
    getCurrentUser() {
        const user = localStorage.getItem(STORAGE_KEYS.USER);
        return user ? JSON.parse(user) : null;
    },

    // Set current user
    setCurrentUser(user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
    },

    // Remove current user
    removeCurrentUser() {
        localStorage.removeItem(STORAGE_KEYS.USER);
    },

    // Logout user
    logout() {
        this.removeToken();
        this.removeCurrentUser();
        window.location.href = '/login';
    }
};

// API Request Utilities
export const api = {
    // Base fetch function with authentication
    async request(endpoint, options = {}) {
        const url = `${API_CONFIG.BASE_URL}${endpoint}`;
        const token = auth.getToken();
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...(token && { Authorization: `Bearer ${token}` })
            }
        };

        const mergedOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers
            }
        };

        try {
            const response = await fetch(url, mergedOptions);
            
            // Handle authentication errors
            if (response.status === 401) {
                auth.logout();
                throw new Error('Authentication required');
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Request failed');
            }

            return await response.json();
        } catch (error) {
            console.error('API Request Error:', error);
            throw error;
        }
    },

    // GET request
    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },

    // POST request
    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    // PUT request
    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    // DELETE request
    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

// Image Utilities
export const images = {
    // Get TMDB image URL
    getTmdbImageUrl(path, size = 'w500') {
        if (!path) return '/images/no-image.jpg';
        return `${API_CONFIG.TMDB_IMAGE_BASE}${size}${path}`;
    },

    // Get poster URL
    getPosterUrl(path) {
        return this.getTmdbImageUrl(path, API_CONFIG.TMDB_IMAGE_SIZES.poster);
    },

    // Get backdrop URL
    getBackdropUrl(path) {
        return this.getTmdbImageUrl(path, API_CONFIG.TMDB_IMAGE_SIZES.backdrop);
    },

    // Get profile URL
    getProfileUrl(path) {
        return this.getTmdbImageUrl(path, API_CONFIG.TMDB_IMAGE_SIZES.profile);
    },

    // Handle image loading errors
    handleImageError(event) {
        event.target.src = '/images/no-image.jpg';
    }
};

// Date Utilities
export const dates = {
    // Format date for display
    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    },

    // Format date for API
    formatDateForAPI(date) {
        if (!date) return null;
        return new Date(date).toISOString().split('T')[0];
    },

    // Get year from date
    getYear(dateString) {
        if (!dateString) return 'Unknown';
        return new Date(dateString).getFullYear();
    },

    // Calculate age from date
    calculateAge(dateString) {
        if (!dateString) return null;
        const birthDate = new Date(dateString);
        const today = new Date();
        return today.getFullYear() - birthDate.getFullYear();
    }
};

// Format Utilities
export const format = {
    // Format runtime in minutes to hours and minutes
    runtime(minutes) {
        if (!minutes) return 'Unknown';
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
    },

    // Format rating
    rating(rating) {
        if (!rating) return 'N/A';
        return Number(rating).toFixed(1);
    },

    // Format currency
    currency(amount, currency = 'USD') {
        if (!amount) return 'Free';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },

    // Format large numbers
    number(num) {
        if (!num) return '0';
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    },

    // Truncate text
    truncate(text, maxLength = 150) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
};

// Genre Utilities
export const genres = {
    // TMDB Genre mapping
    MOVIE_GENRES: {
        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        14: "Fantasy",
        36: "History",
        27: "Horror",
        10402: "Music",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        10770: "TV Movie",
        53: "Thriller",
        10752: "War",
        37: "Western"
    },

    TV_GENRES: {
        10759: "Action & Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        10762: "Kids",
        9648: "Mystery",
        10763: "News",
        10764: "Reality",
        10765: "Sci-Fi & Fantasy",
        10766: "Soap",
        10767: "Talk",
        10768: "War & Politics",
        37: "Western"
    },

    // Get genre name by ID
    getGenreName(id, contentType = 'movie') {
        const genreMap = contentType === 'movie' ? this.MOVIE_GENRES : this.TV_GENRES;
        return genreMap[id] || 'Unknown';
    },

    // Get genre names from array of IDs
    getGenreNames(ids, contentType = 'movie') {
        if (!ids || !Array.isArray(ids)) return [];
        return ids.map(id => this.getGenreName(id, contentType));
    },

    // Format genres for display
    formatGenres(genres, maxGenres = 3) {
        if (!genres || !Array.isArray(genres)) return '';
        const displayGenres = genres.slice(0, maxGenres);
        return displayGenres.join(', ');
    }
};

// Validation Utilities
export const validation = {
    // Validate email
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },

    // Validate password
    isValidPassword(password) {
        return password && password.length >= 6;
    },

    // Validate username
    isValidUsername(username) {
        return username && username.length >= 3 && /^[a-zA-Z0-9_]+$/.test(username);
    },

    // Validate rating
    isValidRating(rating) {
        return rating >= 1 && rating <= 10;
    }
};

// Local Storage Utilities
export const storage = {
    // Save to localStorage with error handling
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Error saving to localStorage:', error);
        }
    },

    // Get from localStorage with error handling
    get(key) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : null;
        } catch (error) {
            console.error('Error reading from localStorage:', error);
            return null;
        }
    },

    // Remove from localStorage
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('Error removing from localStorage:', error);
        }
    },

    // Clear all localStorage
    clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.error('Error clearing localStorage:', error);
        }
    }
};

// Search Utilities
export const search = {
    // Debounce function for search input
    debounce(func, delay = 300) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    },

    // Highlight search terms in text
    highlightText(text, searchTerm) {
        if (!text || !searchTerm) return text;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    },

    // Filter array by search term
    filterBySearch(items, searchTerm, fields = ['title', 'name']) {
        if (!searchTerm) return items;
        const term = searchTerm.toLowerCase();
        return items.filter(item =>
            fields.some(field => 
                item[field] && item[field].toLowerCase().includes(term)
            )
        );
    }
};

// URL Utilities
export const url = {
    // Get query parameter
    getParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    },

    // Set query parameter
    setParam(name, value) {
        const url = new URL(window.location);
        url.searchParams.set(name, value);
        window.history.pushState({}, '', url);
    },

    // Remove query parameter
    removeParam(name) {
        const url = new URL(window.location);
        url.searchParams.delete(name);
        window.history.pushState({}, '', url);
    },

    // Build query string
    buildQuery(params) {
        const query = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                query.append(key, value);
            }
        });
        return query.toString();
    }
};

// Error Handling Utilities
export const errors = {
    // Handle API errors
    handleApiError(error) {
        if (error.message === 'Authentication required') {
            auth.logout();
            return;
        }
        
        // Show error message to user
        this.showError(error.message || 'An error occurred');
    },

    // Show error message
    showError(message) {
        // You can customize this to use your preferred notification system
        console.error(message);
        // Example: toast.error(message);
    },

    // Show success message
    showSuccess(message) {
        // You can customize this to use your preferred notification system
        console.log(message);
        // Example: toast.success(message);
    }
};

// Theme Utilities
export const theme = {
    // Get current theme
    getCurrentTheme() {
        return localStorage.getItem('theme') || 'light';
    },

    // Set theme
    setTheme(theme) {
        localStorage.setItem('theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
    },

    // Toggle theme
    toggleTheme() {
        const currentTheme = this.getCurrentTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
        return newTheme;
    }
};

// Content Utilities
export const content = {
    // Get content type display name
    getContentTypeDisplayName(type) {
        const types = {
            movie: 'Movie',
            tv: 'TV Show',
            anime: 'Anime',
            documentary: 'Documentary'
        };
        return types[type] || 'Unknown';
    },

    // Check if content is movie
    isMovie(content) {
        return content.content_type === 'movie' || content.title;
    },

    // Check if content is TV show
    isTVShow(content) {
        return content.content_type === 'tv' || content.name;
    },

    // Get content title
    getTitle(content) {
        return content.title || content.name || 'Unknown Title';
    },

    // Get content release year
    getReleaseYear(content) {
        const dateString = content.release_date || content.first_air_date;
        return dateString ? new Date(dateString).getFullYear() : 'Unknown';
    }
};

// Animation Utilities
export const animation = {
    // Smooth scroll to element
    scrollToElement(element, offset = 0) {
        const targetPosition = element.offsetTop - offset;
        window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
        });
    },

    // Fade in element
    fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.display = 'block';
        
        const start = performance.now();
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            element.style.opacity = progress;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    },

    // Fade out element
    fadeOut(element, duration = 300) {
        const start = performance.now();
        const startOpacity = parseFloat(element.style.opacity) || 1;
        
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            element.style.opacity = startOpacity * (1 - progress);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.style.display = 'none';
            }
        };
        
        requestAnimationFrame(animate);
    }
};

// Export all utilities
export default {
    auth,
    api,
    images,
    dates,
    format,
    genres,
    validation,
    storage,
    search,
    url,
    errors,
    theme,
    content,
    animation,
    API_CONFIG,
    STORAGE_KEYS
};