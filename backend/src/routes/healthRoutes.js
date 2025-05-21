const express = require('express');
const router = express.Router();
const { performHealthCheck } = require('../utils/dbHealthCheck');

/**
 * @route GET /health
 * @description Basic health check endpoint
 * @access Public
 */
router.get('/', (req, res) => {
  res.status(200).json({ status: 'ok', service: 'SMOOPs API' });
});

/**
 * @route GET /health/db
 * @description Database health check endpoint
 * @access Public
 */
router.get('/db', async (req, res) => {
  try {
    const healthResult = await performHealthCheck();
    
    // Set appropriate status code based on health check result
    const statusCode = healthResult.isHealthy ? 200 : 503;
    
    res.status(statusCode).json(healthResult);
  } catch (error) {
    res.status(500).json({
      isHealthy: false,
      message: 'Health check failed',
      error: error.message
    });
  }
});

module.exports = router; 