const watchlistGrid = document.getElementById('watchlist-grid');

async function loadWatchlist() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (!user) return;

    try {
        const response = await apiRequest('/watchlist');
        displayWatchlist(response.watchlist);
    } catch (error) {
        console.error('Watchlist error:', error);
    }
}

function displayWatchlist(items) {
    watchlistGrid.innerHTML = '';
    items.forEach(item => {
        const card = createContentCard(item);
        watchlistGrid.appendChild(card);
    });
}

async function addToWatchlist(contentId) {
    try {
        await apiRequest('/watchlist', 'POST', { content_id: contentId });
        loadWatchlist();
        alert('Added to watchlist');
    } catch (error) {
        alert('Failed to add to watchlist: ' + error.message);
    }
}

async function removeFromWatchlist(contentId) {
    try {
        await apiRequest(`/watchlist?content_id=${contentId}`, 'DELETE');
        loadWatchlist();
        alert('Removed from watchlist');
    } catch (error) {
        alert('Failed to remove from watchlist: ' + error.message);
    }
}

async function rateContent(contentId, rating) {
    try {
        await apiRequest('/rate', 'POST', { content_id: contentId, rating });
        alert('Rating saved');
    } catch (error) {
        alert('Failed to save rating: ' + error.message);
    }
}