/**
 * Authentication Routes
 * Handles user registration, login, and token management
 */

const express = require('express');
const router = express.Router();
const { register, login, refreshToken } = require('../controllers/authController');
const { verifyRefreshToken } = require('../middleware/auth');

// Register new user
router.post('/register', register);

// Login user
router.post('/login', login);

// Refresh token
router.post('/refresh', verifyRefreshToken, refreshToken);

module.exports = router; 