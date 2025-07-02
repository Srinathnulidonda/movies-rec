const API_BASE_URL = 'https://movies-rec-1hcv.onrender.com/api'; // Update to your backend URL for Vercel deployment
let token = localStorage.getItem('token');
let isAdmin = false;

document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    setupEventListeners();
    const path = window.location.pathname;
    if (path === '/index.html' || path === '/') loadTrending();
    if (path === '/recommendations.html') loadRecommendations();
    if (path === '/wishlist.html') loadWishlist();
    if (path === '/favorites.html') loadFavorites();
    if (path === '/details.html') loadContentDetails();
    if (path === '/admin.html') loadAdminPanel();
});

function checkAuth() {
    if (token) {
        document.getElementById('loginBtn').style.display = 'none';
        document.getElementById('logoutBtn').style.display = 'block';
        fetch(`${API_BASE_URL}/auth/login`, {
            headers: { 'Authorization': `Bearer ${token}` }
        })
            .then(response => response.json())
            .then(data => {
                if (data.user && data.user.is_admin) {
                    isAdmin = true;
                    document.getElementById('adminLink').style.display = 'block';
                }
            })
            .catch(() => {
                localStorage.removeItem('token');
                window.location.href = '/index.html';
            });
    } else {
        document.getElementById('loginBtn').style.display = 'block';
        document.getElementById('logoutBtn').style.display = 'none';
        document.getElementById('adminLink').style.display = 'none';
    }
}

function setupEventListeners() {
    document.getElementById('loginBtn')?.addEventListener('click', () => {
        new bootstrap.Modal(document.getElementById('loginModal')).show();
    });

    document.getElementById('logoutBtn')?.addEventListener('click', () => {
        localStorage.removeItem('token');
        token = null;
        isAdmin = false;
        window.location.reload();
    });

    document.getElementById('showRegister')?.addEventListener('click', () => {
        bootstrap.Modal.getInstance(document.getElementById('loginModal')).hide();
        new bootstrap.Modal(document.getElementById('registerModal')).show();
    });

    document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        try {
            const response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await response.json();
            if (response.ok) {
                token = data.token;
                localStorage.setItem('token', token);
                isAdmin = data.user.is_admin;
                bootstrap.Modal.getInstance(document.getElementById('loginModal')).hide();
                window.location.reload();
            } else {
                alert(data.message);
            }
        } catch (error) {
            alert('Login failed');
        }
    });

    document.getElementById('registerForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('regUsername').value;
        const email = document.getElementById('regEmail').value;
        const password = document.getElementById('regPassword').value;
        try {
            const response = await fetch(`${API_BASE_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password })
            });
            const data = await response.json();
            if (response.ok) {
                token = data.token;
                localStorage.setItem('token', token);
                bootstrap.Modal.getInstance(document.getElementById('registerModal')).hide();
                window.location.reload();
            } else {
                alert(data.message);
            }
        } catch (error) {
            alert('Registration failed');
        }
    });

    document.getElementById('searchBtn')?.addEventListener('click', async () => {
        const query = document.getElementById('searchInput').value;
        const contentType = document.getElementById('contentType').value;
        if (query) {
            try {
                const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}&type=${contentType}`);
                const data = await response.json();
                if (response.ok) {
                    displaySearchResults(data.results);
                } else {
                    alert(data.message);
                }
            } catch (error) {
                alert('Search failed');
            }
        }
    });

    document.getElementById('movieSearch')?.addEventListener('input', async (e) => {
        const query = e.target.value;
        if (query.length > 2) {
            try {
                const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}&type=multi`);
                const data = await response.json();
                if (response.ok) {
                    const select = document.getElementById('movieSelect');
                    select.innerHTML = '<option value="">Select a movie</option>';
                    data.results.forEach(item => {
                        const option = document.createElement('option');
                        option.value = item.id;
                        option.textContent = item.title;
                        select.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Movie search failed:', error);
            }
        }
    });

    document.getElementById('suggestionForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!isAdmin) return alert('Admin access required');
        const movieId = document.getElementById('movieSelect').value;
        const title = document.getElementById('suggestionTitle').value;
        const description = document.getElementById('suggestionDescription').value;
        try {
            const response = await fetch(`${API_BASE_URL}/admin/suggestions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ movie_id: movieId, title, description })
            });
            const data = await response.json();
            if (response.ok) {
                alert('Suggestion posted successfully');
                document.getElementById('suggestionForm').reset();
                loadAdminPanel();
            } else {
                alert(data.message);
            }
        } catch (error) {
            alert('Failed to post suggestion');
        }
    });
}

async function loadTrending() {
    try {
        const response = await fetch(`${API_BASE_URL}/trending?type=all`);
        const data = await response.json();
        if (response.ok) {
            displaySearchResults(data.trending);
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load trending content');
    }
}

async function loadRecommendations() {
    if (!token) return alert('Please login to view recommendations');
    try {
        const response = await fetch(`${API_BASE_URL}/recommendations`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
            displaySearchResults(data.recommendations);
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load recommendations');
    }
}

async function loadWishlist() {
    if (!token) return alert('Please login to view wishlist');
    try {
        const response = await fetch(`${API_BASE_URL}/watchlist`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
            displaySearchResults(data.watchlist);
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load wishlist');
    }
}

async function loadFavorites() {
    if (!token) return alert('Please login to view favorites');
    try {
        const response = await fetch(`${API_BASE_URL}/favorites`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
            displaySearchResults(data.favorites);
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load favorites');
    }
}

async function loadContentDetails() {
    const urlParams = new URLSearchParams(window.location.search);
    const contentId = urlParams.get('id');
    if (!contentId) return alert('Content ID missing');
    try {
        const response = await fetch(`${API_BASE_URL}/content/${contentId}`);
        const data = await response.json();
        if (response.ok) {
            displayContentDetails(data);
            displaySearchResults(data.similar, 'similarContent');
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load content details');
    }
}

async function loadAdminPanel() {
    if (!isAdmin) return alert('Admin access required');
    try {
        const response = await fetch(`${API_BASE_URL}/admin/suggestions`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        if (response.ok) {
            displaySearchResults(data.suggestions, 'suggestionsList');
        } else {
            alert(data.message);
        }
    } catch (error) {
        alert('Failed to load admin panel');
    }
}

function displaySearchResults(items, containerId = 'trendingList') {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'col-md-3 mb-4';
        div.innerHTML = `
            <div class="card movie-card">
                <img src="${item.poster_path || 'https://via.placeholder.com/300x450'}" class="card-img-top" alt="${item.title}">
                <div class="card-body">
                    <h5 class="card-title">${item.title}</h5>
                    <p class="card-text">${item.overview ? item.overview.substring(0, 100) + '...' : ''}</p>
                    <a href="/details.html?id=${item.id}" class="btn btn-primary btn-sm">View Details</a>
                    ${token ? `
                        <button class="btn btn-outline-primary btn-sm mt-2" onclick="addToWishlist(${item.id})">Add to Wishlist</button>
                        <button class="btn btn-outline-primary btn-sm mt-2" onclick="addToFavorites(${item.id})">Add to Favorites</button>
                    ` : ''}
                </div>
            </div>
        `;
        container.appendChild(div);
    });
}

function displayContentDetails(data) {
    const container = document.getElementById('contentDetails');
    container.innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <img src="${data.poster_path || 'https://via.placeholder.com/300x450'}" class="movie-details w-100" alt="${data.title}">
            </div>
            <div class="col-md-8">
                <h1 class="text-2xl font-bold">${data.title}</h1>
                <p class="text-muted">${data.tagline}</p>
                <p><strong>Release Date:</strong> ${data.release_date}</p>
                <p><strong>Rating:</strong> ${data.vote_average}/10 (${data.vote_count} votes)</p>
                <p><strong>Runtime:</strong> ${data.runtime ? `${data.runtime} min` : 'N/A'}</p>
                <p><strong>Genres:</strong> ${data.genres.map(g => g.name).join(', ')}</p>
                <p><strong>Overview:</strong> ${data.overview}</p>
                <p><strong>Cast:</strong> ${data.cast.map(c => c.name).join(', ')}</p>
                <h3 class="mt-4">Streaming Platforms</h3>
                <ul>
                    ${data.streaming_platforms.map(sp => `
                        <li>
                            ${sp.platform_name} (${sp.stream_type}, ${sp.quality})
                            <a href="${sp.url}" target="_blank" class="text-blue-500">Watch Now</a>
                        </li>
                    `).join('')}
                </ul>
                ${data.videos.length ? `
                    <h3 class="mt-4">Videos</h3>
                    ${data.videos.map(v => `
                        <a href="https://www.youtube.com/watch?v=${v.key}" target="_blank" class="text-blue-500">${v.name}</a><br>
                    `).join('')}
                ` : ''}
            </div>
        </div>
    `;
}

async function addToWishlist(movieId) {
    if (!token) return alert('Please login to add to wishlist');
    try {
        const response = await fetch(`${API_BASE_URL}/watchlist`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ movie_id: movieId })
        });
        const data = await response.json();
        alert(data.message);
    } catch (error) {
        alert('Failed to add to wishlist');
    }
}

async function addToFavorites(movieId) {
    if (!token) return alert('Please login to add to favorites');
    try {
        const response = await fetch(`${API_BASE_URL}/favorites`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ movie_id: movieId })
        });
        const data = await response.json();
        alert(data.message);
    } catch (error) {
        alert('Failed to add to favorites');
    }
}