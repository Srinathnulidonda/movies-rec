const searchInput = document.getElementById('search-input');
const suggestionsContainer = document.getElementById('search-suggestions');

async function fetchSearchResults(query) {
    try {
        const results = await apiRequest(`/search?q=${encodeURIComponent(query)}`);
        displaySearchSuggestions(results);
    } catch (error) {
        console.error('Search error:', error);
    }
}

function displaySearchSuggestions(results) {
    suggestionsContainer.innerHTML = '';
    const suggestions = results.suggestions || [];
    suggestions.forEach(item => {
        const div = document.createElement('div');
        div.className = 'px-4 py-2 hover:bg-[#30363d] cursor-pointer';
        div.textContent = item.title;
        div.addEventListener('click', () => showContentDetails(item.id));
        suggestionsContainer.appendChild(div);
    });
    suggestionsContainer.classList.toggle('hidden', suggestions.length === 0);
}

searchInput.addEventListener('input', debounce(async () => {
    const query = searchInput.value.trim();
    if (query.length > 2) {
        await fetchSearchResults(query);
    } else {
        suggestionsContainer.classList.add('hidden');
    }
}, 300));

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