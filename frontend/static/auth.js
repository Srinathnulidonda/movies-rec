// Authentication utilities
class AuthManager {
    constructor() {
        this.token = localStorage.getItem(CONFIG.STORAGE_KEYS.TOKEN);
        this.userId = localStorage.getItem(CONFIG.STORAGE_KEYS.USER_ID);
        this.updateUI();
    }

    isAuthenticated() {
        return !!this.token;
    }

    getToken() {
        return this.token;
    }

    getUserId() {
        return this.userId;
    }

    setAuth(token, userId) {
        this.token = token;
        this.userId = userId;
        localStorage.setItem(CONFIG.STORAGE_KEYS.TOKEN, token);
        localStorage.setItem(CONFIG.STORAGE_KEYS.USER_ID, userId);
        this.updateUI();
    }

    logout() {
        this.token = null;
        this.userId = null;
        localStorage.removeItem(CONFIG.STORAGE_KEYS.TOKEN);
        localStorage.removeItem(CONFIG.STORAGE_KEYS.USER_ID);
        this.clearCache();
        this.updateUI();
        window.location.href = 'index.html';
    }

    clearCache() {
        Object.keys(localStorage).forEach(key => {
            if (key.startsWith(CONFIG.STORAGE_KEYS.CACHE_PREFIX)) {
                localStorage.removeItem(key);
            }
        });
    }

    updateUI() {
        const authButtons = document.getElementById('authButtons');
        const guestButtons = document.getElementById('guestButtons');
        const recommendationsSection = document.getElementById('recommendationsSection');

        if (this.isAuthenticated()) {
            if (authButtons) authButtons.classList.remove('d-none');
            if (guestButtons) guestButtons.classList.add('d-none');
            if (recommendationsSection) recommendationsSection.classList.remove('d-none');
        } else {
            if (authButtons) authButtons.classList.add('d-none');
            if (guestButtons) guestButtons.classList.remove('d-none');
            if (recommendationsSection) recommendationsSection.classList.add('d-none');
        }
    }

    getAuthHeaders() {
        return this.token ? {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json'
        } : {
            'Content-Type': 'application/json'
        };
    }

    async register(username, email, password) {
        try {
            const response = await fetch(getApiUrl('REGISTER'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password })
            });

            if (response.ok) {
                const data = await response.json();
                this.setAuth(data.access_token, data.user_id);
                return { success: true, data };
            } else {
                const error = await response.json();
                return { success: false, error: error.error || 'Registration failed' };
            }
        } catch (error) {
            return { success: false, error: CONFIG.ERRORS.NETWORK };
        }
    }

    async login(username, password) {
        try {
            const response = await fetch(getApiUrl('LOGIN'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            if (response.ok) {
                const data = await response.json();
                this.setAuth(data.access_token, data.user_id);
                return { success: true, data };
            } else {
                const error = await response.json();
                return { success: false, error: error.error || 'Login failed' };
            }
        } catch (error) {
            return { success: false, error: CONFIG.ERRORS.NETWORK };
        }
    }
}

// API utility class
class ApiClient {
    constructor(authManager) {
        this.auth = authManager;
    }

    async request(url, options = {}) {
        const config = {
            headers: this.auth.getAuthHeaders(),
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (response.status === 401) {
                this.auth.logout();
                throw new Error(CONFIG.ERRORS.UNAUTHORIZED);
            }

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    async get(endpoint, params = {}) {
        const url = new URL(getApiUrl(endpoint));
        Object.keys(params).forEach(key => {
            if (params[key] !== undefined && params[key] !== null) {
                url.searchParams.append(key, params[key]);
            }
        });
        
        return this.request(url.toString());
    }

    async post(endpoint, data) {
        return this.request(getApiUrl(endpoint), {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint, params = {}) {
        const url = new URL(getApiUrl(endpoint));
        Object.keys(params).forEach(key => {
            if (params[key] !== undefined && params[key] !== null) {
                url.searchParams.append(key, params[key]);
            }
        });
        
        return this.request(url.toString(), {
            method: 'DELETE'
        });
    }
}

// Initialize auth manager
const authManager = new AuthManager();
const apiClient = new ApiClient(authManager);

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            authManager.logout();
        });
    }
});

// Utility functions
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.style.position = 'fixed';
    toast.style.top = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast, { delay: CONFIG.UI.TOAST_DURATION });
    bsToast.show();

    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function showLoading(show = true) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        if (show) {
            spinner.classList.remove('d-none');
        } else {
            spinner.classList.add('d-none');
        }
    }
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function truncateText(text, maxLength = 150) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}