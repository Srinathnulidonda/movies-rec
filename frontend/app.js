// app.js
import { AuthManager } from './auth.js';
import { APIService } from './api.js';
import { ContentManager } from './content.js';
import { PageManager } from './pages.js';
import { SearchManager } from './search.js';
import { WatchlistManager } from './watchlist.js';
import { ReviewManager } from './reviews.js';

// Initialize Managers
const authManager = new AuthManager();
const pageManager = new PageManager();
const searchManager = new SearchManager();
const watchlistManager = new WatchlistManager();
const reviewManager = new ReviewManager();

// Global Variables
let deferredPrompt;

// Utility Functions
const showLoading = () => {
    document.getElementById('loadingSpinner').style.display = 'flex';
};

const hideLoading = () => {
    document.getElementById('loadingSpinner').style.display = 'none';
};

const showToast = (message, type = 'info') => {
    const toast = document.getElementById('toastNotification');
    const toastBody = document.getElementById('toastBody');
    toastBody.textContent = message;
    toast.className = `toast show bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'}`;
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
};

// Theme Toggle
const toggleTheme = () => {
    document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
};

// PWA Install Prompt
const installApp = async () => {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        if (outcome === 'accepted') {
            showToast('App installed successfully', 'success');
        }
        deferredPrompt = null;
        document.getElementById('installPrompt').style.display = 'none';
    }
};

// Global Event Handlers
window.showHomePage = () => pageManager.showHomePage();
window.showMovies = () => pageManager.showMoviesPage();
window.showTVShows = () => pageManager.showTVShowsPage();
window.showAnime = () => pageManager.showAnimePage();
window.showWatchlist = () => pageManager.showWatchlistPage();
window.showContentDetails = (mediaType, id) => pageManager.showDetailPage(mediaType, id);
window.showLogin = () => authManager.showLogin();
window.showRegister = () => authManager.showRegister();
window.logout = () => authManager.logout();
window.addToWatchlist = () => watchlistManager.addToWatchlist();
window.toggleFavorite = () => watchlistManager.toggleFavorite();
window.shareContent = () => watchlistManager.shareContent();
window.playTrailer = () => pageManager.playTrailer();
window.playVideo = (videoKey, title) => pageManager.playVideo(videoKey, title);
window.showReviewModal = () => reviewManager.showReviewModal();
window.switchRegion = (region) => pageManager.switchRegion(region);
window.toggleTheme = toggleTheme;
window.installApp = installApp;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    // Load theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
    }

    // Initialize forms
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        await authManager.login(email, password);
    });

    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('registerName').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        await authManager.register(name, email, password);
    });

    document.getElementById('reviewForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const rating = document.querySelector('.star-rating input:checked')?.value || 0;
        const review = document.getElementById('reviewText').value;
        await reviewManager.addReview(rating, review);
        bootstrap.Modal.getInstance(document.getElementById('reviewModal')).hide();
    });

    // Star rating
    document.querySelectorAll('.star-rating label').forEach(label => {
        label.addEventListener('click', () => {
            const rating = label.dataset.rating;
            document.querySelector(`#star${rating}`).checked = true;
        });
    });

    // Genre filter
    document.getElementById('genreFilter').addEventListener('change', () => {
        const genre = document.getElementById('genreFilter').value;
        if (genre) {
            pageManager.loadGenreContent(genre);
        }
    });

    // Load initial content
    pageManager.showHomePage();

    // PWA Install Prompt
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        document.getElementById('installPrompt').style.display = 'block';
    });

    // Service Worker Registration
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').then(registration => {
            console.log('ServiceWorker registered:', registration);
        }).catch(err => {
            console.error('ServiceWorker registration failed:', err);
        });
    }

    // Responsive Design
    const checkScreenSize = () => {
        document.body.classList.toggle('mobile-view', window.innerWidth <= 768);
    };
    window.addEventListener('resize', checkScreenSize);
    checkScreenSize();
});