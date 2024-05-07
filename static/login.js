// Get the login and signup forms
const loginForm = document.querySelector('.login-form form');
const signupForm = document.querySelector('.signup-form form');

// Add event listener to login form
loginForm.addEventListener('submit', (e) => {
  e.preventDefault(); // prevent the default form submission
  const email = loginForm.querySelector('input[type="email"]').value;
  const password = loginForm.querySelector('input[type="password"]').value;
  // TODO: validate email and password
  // TODO: send login request to server
});

// Add event listener to signup form
signupForm.addEventListener('submit', (e) => {
  e.preventDefault(); // prevent the default form submission
  const fullName = signupForm.querySelector('input[type="text"]').value;
  const email = signupForm.querySelector('input[type="email"]').value;
  const password = signupForm.querySelector('input[type="password"]').value;
  // TODO: validate full name, email, and password
  // TODO: send signup request to server
});