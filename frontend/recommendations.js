const recommendationsGrid = document.getElementById('recommendations-grid');
const algorithmSelect = document.getElementById('algorithm-select');
const personalizationScore = document.getElementById('personalization-score');
const diversityScore = document.getElementById('diversity-score');

async function loadRecommendations() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (!user) return;

    try {
        const algorithm = algorithmSelect.value;
        const response = await apiRequest(`/recommendations?algorithm=${algorithm}`);
        displayRecommendations(response.recommendations);
        personalizationScore.textContent = response.personalization_score.toFixed(2);
        diversityScore.textContent = response.diversity_score.toFixed(2);
    } catch (error) {
        console.error('Recommendations error:', error);
    }
}

function displayRecommendations(items) {
    recommendationsGrid.innerHTML = '';
    items.forEach(item => {
        const card = createContentCard(item);
        recommendationsGrid.appendChild(card);
    });
}

algorithmSelect.addEventListener('change', loadRecommendations);