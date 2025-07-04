const contentModal = document.getElementById('content-modal');
const modalTitle = document.getElementById('modal-title');
const modalContent = document.getElementById('modal-content');
const modalClose = document.getElementById('modal-close');
const modalWatchlist = document.getElementById('modal-watchlist');
const modalRating = document.getElementById('modal-rating');
const trendingCarousel = document.getElementById('trending-carousel');

async function loadTrending() {
    try {
        const response = await apiRequest('/trending');
        const swiperWrapper = trendingCarousel.querySelector('.swiper-wrapper');
        swiperWrapper.innerHTML = '';
        ['movies', 'tv', 'anime'].forEach(type => {
            response[type].forEach(item => {
                const slide = document.createElement('div');
                slide.className = 'swiper-slide';
                slide.appendChild(createContentCard(item));
                swiperWrapper.appendChild(slide);
            });
        });
        new Swiper(trendingCarousel, {
            slidesPerView: 1,
            spaceBetween: 10,
            navigation: {
                nextEl: '.swiper-button-next',
                prevEl: '.swiper-button-prev',
            },
            breakpoints: {
                640: { slidesPerView: 2 },
                768: { slidesPerView: 3 },
                1024: { slidesPerView: 4 },
            }
        });
    } catch (error) {
        console.error('Trending error:', error);
    }
}

async function loadFeatured() {
    try {
        const response = await apiRequest('/featured');
        const featuredGrid = document.getElementById('featured-grid');
        featuredGrid.innerHTML = '';
        response.featured.forEach(item => {
            const card = createContentCard(item);
            featuredGrid.appendChild(card);
        });
    } catch (error) {
        console.error('Featured error:', error);
    }
}

function createContentCard(item) {
    const card = document.createElement('div');
    card.className = 'content-card';
    card.innerHTML = `
        <img src="${item.poster_path || '/assets/images/placeholder.jpg'}" alt="${item.title}" class="lazy">
        <div class="p-4">
            <h3 class="text-lg font-semibold">${item.title}</h3>
            <div class="flex items-center space-x-2">
                <span class="text-[#f0b90b]"><i class="fas fa-star"></i> ${item.vote_average?.toFixed(1) || 'N/A'}</span>
                <span class="text-[#8b949e]">${item.content_type}</span>
            </div>
            <button class="view-details btn bg-[#58a6ff] hover:bg-[#f0b90b] text-[#0d1117] mt-2 px-4 py-1 rounded" data-id="${item.id}">View Details</button>
        </div>
    `;
    card.querySelector('.view-details').addEventListener('click', () => showContentDetails(item.id));
    return card;
}

async function showContentDetails(contentId) {
    try {
        const content = await apiRequest(`/content/${contentId}`);
        modalTitle.textContent = content.title;
        modalContent.innerHTML = `
            <img src="${content.poster_path || '/assets/images/placeholder.jpg'}" alt="${content.title}" class="w-full h-64 object-cover rounded mb-4">
            <p>${content.overview || 'No description available'}</p>
            <div class="mt-4">
                <p><strong>Release Date:</strong> ${content.release_date || 'N/A'}</p>
                <p><strong>Rating:</strong> ${content.vote_average?.toFixed(1) || 'N/A'}</p>
                <p><strong>Genres:</strong> ${JSON.parse(content.genre_ids || '[]').join(', ') || 'N/A'}</p>
            </div>
            <div class="mt-4">
                <h4 class="font-semibold">Similar Content</h4>
                <div class="grid grid-cols-2 gap-4">
                    ${content.similar_content.map(item => `
                        <div class="content-card">
                            <img src="${item.poster_path || '/assets/images/placeholder.jpg'}" alt="${item.title}" class="w-full h-32 object-cover">
                            <p class="p-2 text-sm">${item.title}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        modalWatchlist.onclick = () => content.user_interactions?.watchlist ? removeFromWatchlist(contentId) : addToWatchlist(contentId);
        modalWatchlist.textContent = content.user_interactions?.watchlist ? 'Remove from Watchlist' : 'Add to Watchlist';
        modalRating.value = content.user_interactions?.rating || '';
        modalRating.onchange = () => rateContent(contentId, modalRating.value);
        contentModal.classList.remove('hidden');
    } catch (error) {
        console.error('Content details error:', error);
    }
}

modalClose.addEventListener('click', () => contentModal.classList.add('hidden'));

// Lazy Loading
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src || img.src;
            observer.unobserve(img);
        }
    });
}, { rootMargin: '0px 0px 100px 0px' });

document.querySelectorAll('img.lazy').forEach(img => observer.observe(img));

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadTrending();
    loadFeatured();
    loadRecommendations();
    loadWatchlist();
});