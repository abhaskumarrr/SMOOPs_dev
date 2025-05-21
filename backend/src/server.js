/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config({
  path: path.resolve(__dirname, '../../.env')
});

// Import routes
const apiKeyRoutes = require('./routes/apiKeyRoutes');
const deltaApiRoutes = require('./routes/deltaApiRoutes');
const healthRoutes = require('./routes/healthRoutes');

// Create Express app
const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet()); // Security headers
app.use(cors()); // Enable CORS for frontend
app.use(express.json()); // Parse JSON request bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies

// Routes
app.use('/health', healthRoutes);
app.use('/api/keys', apiKeyRoutes);
app.use('/api/delta', deltaApiRoutes);

// Global error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    message: 'An unexpected error occurred',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;

