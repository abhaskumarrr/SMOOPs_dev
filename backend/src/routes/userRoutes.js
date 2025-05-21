/**
 * User Routes
 * Endpoints for user profile management
 */

const express = require('express');
const router = express.Router();
const { getProfile, updateProfile, getUsers } = require('../controllers/userController');
const { protect } = require('../middleware/auth');

// Get current user profile
router.get('/profile', protect, getProfile);

// Update current user profile
router.put('/profile', protect, updateProfile);

// Admin only: Get all users
router.get('/', protect, getUsers);

module.exports = router; 