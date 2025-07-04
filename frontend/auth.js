const loginModal = document.getElementById('login-modal');
const registerModal = document.getElementById('register-modal');
const loginBtn = document.getElementById('login-btn');
const authSection = document.getElementById('auth-section');

function updateAuthUI() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (user) {
        authSection.innerHTML = `
            <div class="relative">
                <button id="profile-btn" class="flex items-center space-x-2">
                    <span>${user.username}</span>
                    <i class="fas fa-user-circle"></i>
                </button>
                <div id="profile-dropdown" class="hidden absolute right-0 mt-2 w-48 bg-[#21262d] rounded shadow-lg">
                    <a href="#" id="logout-btn" class="block px-4 py-2 hover:bg-[#30363d]">Logout</a>
                </div>
            </div>
        `;
        document.getElementById('logout-btn').addEventListener('click', logout);
        document.getElementById('profile-btn').addEventListener('click', () => {
            document.getElementById('profile-dropdown').classList.toggle('hidden');
        });
    }
}

async function login() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    try {
        const response = await apiRequest('/auth/login', 'POST', { username, password });
        localStorage.setItem('user', JSON.stringify(response.user));
        loginModal.classList.add('hidden');
        updateAuthUI();
        loadRecommendations();
        loadWatchlist();
    } catch (error) {
        alert('Login failed: ' + error.message);
    }
}

async function register() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    try {
        await apiRequest('/auth/register', 'POST', { username, email, password });
        registerModal.classList.add('hidden');
        alert('Registration successful! Please login.');
        showLoginModal();
    } catch (error) {
        alert('Registration failed: ' + error.message);
    }
}

async function logout() {
    try {
        await apiRequest('/auth/logout', 'POST');
        localStorage.removeItem('user');
        authSection.innerHTML = '<button id="login-btn" class="btn btn-primary bg-[#f0b90b] hover:bg-[#d29922] text-[#0d1117] px-4 py-2 rounded">Login</button>';
        loginBtn = document.getElementById('login-btn');
        loginBtn.addEventListener('click', showLoginModal);
    } catch (error) {
        alert('Logout failed: ' + error.message);
    }
}

function showLoginModal() {
    loginModal.classList.remove('hidden');
    registerModal.classList.add('hidden');
}

function showRegisterModal() {
    registerModal.classList.remove('hidden');
    loginModal.classList.add('hidden');
}

document.getElementById('login-btn').addEventListener('click', showLoginModal);
document.getElementById('login-submit').addEventListener('click', login);
document.getElementById('register-submit').addEventListener('click', register);
document.getElementById('show-register').addEventListener('click', showRegisterModal);
document.getElementById('show-login').addEventListener('click', showLoginModal);

updateAuthUI();