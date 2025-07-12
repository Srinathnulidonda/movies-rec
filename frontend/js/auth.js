// auth.js - Authentication handling for Movie Recommendation App

class AuthManager {
    constructor() {
        this.baseURL = this.getBaseURL();
        this.token = this.getStoredToken();
        this.user = this.getStoredUser();
        this.authCallbacks = [];
    }

    // Get the appropriate base URL based on environment
    getBaseURL() {
        // Check if running on local development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://127.0.0.1:5000';
        }
        // Production backend URL
        return 'https://backend-app-970m.onrender.com';
    }

    // Token management
    getStoredToken() {
        return localStorage.getItem('auth_token');
    }

    setToken(token) {
        this.token = token;
        if (token) {
            localStorage.setItem('auth_token', token);
        } else {
            localStorage.removeItem('auth_token');
        }
    }

    // User management
    getStoredUser() {
        const userData = localStorage.getItem('user_data');
        return userData ? JSON.parse(userData) : null;
    }

    setUser(user) {
        this.user = user;
        if (user) {
            localStorage.setItem('user_data', JSON.stringify(user));
        } else {
            localStorage.removeItem('user_data');
        }
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!(this.token && this.user);
    }

    // Get current user
    getCurrentUser() {
        return this.user;
    }

    // Add authentication state change callback
    onAuthStateChange(callback) {
        this.authCallbacks.push(callback);
    }

    // Notify all callbacks about auth state change
    notifyAuthStateChange() {
        this.authCallbacks.forEach(callback => {
            callback(this.isAuthenticated(), this.user);
        });
    }

    // Make authenticated API requests
    async makeAuthenticatedRequest(url, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const requestOptions = {
            ...options,
            headers
        };

        try {
            const response = await fetch(url, requestOptions);
            
            // Handle token expiration
            if (response.status === 401) {
                this.logout();
                throw new Error('Session expired. Please login again.');
            }

            return response;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    // Register new user
    async register(username, email, password) {
        try {
            const response = await fetch(`${this.baseURL}/api/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username.trim(),
                    email: email.trim(),
                    password: password
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Registration failed');
            }

            // Store auth data
            this.setToken(data.token);
            this.setUser({
                id: data.user_id,
                username: data.username,
                email: email
            });

            this.notifyAuthStateChange();
            return { success: true, user: this.user };

        } catch (error) {
            console.error('Registration error:', error);
            throw error;
        }
    }

    // Login user
    async login(username, password) {
        try {
            const response = await fetch(`${this.baseURL}/api/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username.trim(),
                    password: password
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Login failed');
            }

            // Store auth data
            this.setToken(data.token);
            this.setUser({
                id: data.user_id,
                username: data.username
            });

            this.notifyAuthStateChange();
            return { success: true, user: this.user };

        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    }

    // Logout user
    logout() {
        this.setToken(null);
        this.setUser(null);
        this.notifyAuthStateChange();
        
        // Redirect to login page if needed
        if (window.location.pathname !== '/login.html' && window.location.pathname !== '/index.html') {
            window.location.href = '/login.html';
        }
    }

    // Validate token on page load
    async validateToken() {
        if (!this.token) {
            return false;
        }

        try {
            const response = await this.makeAuthenticatedRequest(`${this.baseURL}/api/recommendations`);
            
            if (response.ok) {
                return true;
            } else {
                this.logout();
                return false;
            }
        } catch (error) {
            console.error('Token validation failed:', error);
            this.logout();
            return false;
        }
    }

    // Get user recommendations (protected route)
    async getRecommendations() {
        try {
            const response = await this.makeAuthenticatedRequest(`${this.baseURL}/api/recommendations`);
            
            if (!response.ok) {
                throw new Error('Failed to fetch recommendations');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            throw error;
        }
    }

    // Record user interaction (protected route)
    async recordInteraction(contentId, interactionType, rating = null) {
        try {
            const response = await this.makeAuthenticatedRequest(`${this.baseURL}/api/interact`, {
                method: 'POST',
                body: JSON.stringify({
                    content_id: contentId,
                    interaction_type: interactionType,
                    rating: rating
                })
            });

            if (!response.ok) {
                throw new Error('Failed to record interaction');
            }

            return await response.json();
        } catch (error) {
            console.error('Error recording interaction:', error);
            throw error;
        }
    }

    // Test API connection
    async testConnection() {
        try {
            const response = await fetch(`${this.baseURL}/api/test`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();
            return { success: response.ok, data };
        } catch (error) {
            console.error('Connection test failed:', error);
            return { success: false, error: error.message };
        }
    }
}

// Create global auth manager instance
const authManager = new AuthManager();

// Authentication utilities
const AuthUtils = {
    // Show loading state
    showLoading(element) {
        if (element) {
            element.disabled = true;
            element.innerHTML = `
                <span class="loading-spinner"></span>
                Loading...
            `;
        }
    },

    // Hide loading state
    hideLoading(element, originalText) {
        if (element) {
            element.disabled = false;
            element.innerHTML = originalText;
        }
    },

    // Show error message
    showError(message, containerId = 'error-container') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    ${message}
                </div>
            `;
            container.style.display = 'block';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                container.style.display = 'none';
            }, 5000);
        }
    },

    // Show success message
    showSuccess(message, containerId = 'success-container') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="success-message">
                    <i class="fas fa-check-circle"></i>
                    ${message}
                </div>
            `;
            container.style.display = 'block';
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                container.style.display = 'none';
            }, 3000);
        }
    },

    // Validate email format
    validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },

    // Validate password strength
    validatePassword(password) {
        return {
            isValid: password.length >= 6,
            message: password.length >= 6 ? '' : 'Password must be at least 6 characters long'
        };
    },

    // Format validation errors
    formatValidationErrors(errors) {
        if (Array.isArray(errors)) {
            return errors.join('<br>');
        }
        return errors;
    },

    // Redirect after login
    redirectAfterLogin() {
        const urlParams = new URLSearchParams(window.location.search);
        const redirect = urlParams.get('redirect');
        
        if (redirect) {
            window.location.href = decodeURIComponent(redirect);
        } else {
            window.location.href = '/dashboard.html';
        }
    },

    // Require authentication for page
    requireAuth() {
        if (!authManager.isAuthenticated()) {
            const currentPath = window.location.pathname + window.location.search;
            window.location.href = `/login.html?redirect=${encodeURIComponent(currentPath)}`;
            return false;
        }
        return true;
    },

    // Initialize auth state on page load
    async initializeAuth() {
        try {
            // Test connection
            const connectionTest = await authManager.testConnection();
            if (!connectionTest.success) {
                console.warn('Backend connection failed:', connectionTest.error);
            }

            // Validate existing token
            if (authManager.token) {
                const isValid = await authManager.validateToken();
                if (!isValid) {
                    console.log('Token validation failed, user logged out');
                }
            }

            return authManager.isAuthenticated();
        } catch (error) {
            console.error('Auth initialization failed:', error);
            return false;
        }
    }
};

// Page-specific auth handlers
const AuthHandlers = {
    // Login page handler
    setupLoginPage() {
        const loginForm = document.getElementById('login-form');
        const loginBtn = document.getElementById('login-btn');
        
        if (loginForm) {
            loginForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                
                // Validation
                if (!username || !password) {
                    AuthUtils.showError('Please fill in all fields');
                    return;
                }
                
                // Show loading
                AuthUtils.showLoading(loginBtn);
                
                try {
                    await authManager.login(username, password);
                    AuthUtils.showSuccess('Login successful! Redirecting...');
                    
                    setTimeout(() => {
                        AuthUtils.redirectAfterLogin();
                    }, 1000);
                    
                } catch (error) {
                    AuthUtils.showError(error.message);
                } finally {
                    AuthUtils.hideLoading(loginBtn, 'Login');
                }
            });
        }
    },

    // Register page handler
    setupRegisterPage() {
        const registerForm = document.getElementById('register-form');
        const registerBtn = document.getElementById('register-btn');
        
        if (registerForm) {
            registerForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const username = document.getElementById('username').value;
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                const confirmPassword = document.getElementById('confirm-password').value;
                
                // Validation
                if (!username || !email || !password || !confirmPassword) {
                    AuthUtils.showError('Please fill in all fields');
                    return;
                }
                
                if (!AuthUtils.validateEmail(email)) {
                    AuthUtils.showError('Please enter a valid email address');
                    return;
                }
                
                const passwordValidation = AuthUtils.validatePassword(password);
                if (!passwordValidation.isValid) {
                    AuthUtils.showError(passwordValidation.message);
                    return;
                }
                
                if (password !== confirmPassword) {
                    AuthUtils.showError('Passwords do not match');
                    return;
                }
                
                // Show loading
                AuthUtils.showLoading(registerBtn);
                
                try {
                    await authManager.register(username, email, password);
                    AuthUtils.showSuccess('Registration successful! Redirecting...');
                    
                    setTimeout(() => {
                        AuthUtils.redirectAfterLogin();
                    }, 1000);
                    
                } catch (error) {
                    AuthUtils.showError(error.message);
                } finally {
                    AuthUtils.hideLoading(registerBtn, 'Register');
                }
            });
        }
    },

    // Setup logout buttons
    setupLogoutButtons() {
        const logoutBtns = document.querySelectorAll('.logout-btn');
        logoutBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                
                if (confirm('Are you sure you want to logout?')) {
                    authManager.logout();
                }
            });
        });
    },

    // Setup user info display
    setupUserInfo() {
        const userInfo = document.querySelector('.user-info');
        const user = authManager.getCurrentUser();
        
        if (userInfo && user) {
            userInfo.innerHTML = `
                <div class="user-details">
                    <span class="username">${user.username}</span>
                    <button class="logout-btn" title="Logout">
                        <i class="fas fa-sign-out-alt"></i>
                    </button>
                </div>
            `;
            
            // Setup logout button
            this.setupLogoutButtons();
        }
    }
};

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize auth
    await AuthUtils.initializeAuth();
    
    // Setup page-specific handlers based on current page
    const currentPage = window.location.pathname.split('/').pop();
    
    switch (currentPage) {
        case 'login.html':
            AuthHandlers.setupLoginPage();
            break;
        case 'register.html':
            AuthHandlers.setupRegisterPage();
            break;
        default:
            // For protected pages
            AuthHandlers.setupLogoutButtons();
            AuthHandlers.setupUserInfo();
            break;
    }
});

// Export for use in other modules
window.authManager = authManager;
window.AuthUtils = AuthUtils;
window.AuthHandlers = AuthHandlers;