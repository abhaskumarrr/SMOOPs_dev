/**
 * Delta Exchange API Routes
 * Endpoints for interacting with the Delta Exchange API
 */

const express = require('express');
const router = express.Router();
const { 
  getProducts,
  getOrderBook,
  getRecentTrades,
  getAccountBalance,
  getPositions,
  createOrder,
  cancelOrder,
  getOrders,
  getMarketData
} = require('../controllers/deltaApiController');
const { protect } = require('../middleware/auth');

// All routes require authentication
router.use(protect);

// Market data endpoints
router.get('/products', getProducts);
router.get('/products/:id/orderbook', getOrderBook);
router.get('/products/:id/trades', getRecentTrades);
router.get('/market-data', getMarketData);

// Account endpoints
router.get('/balance', getAccountBalance);
router.get('/positions', getPositions);

// Order endpoints
router.get('/orders', getOrders);
router.post('/orders', createOrder);
router.delete('/orders/:id', cancelOrder);

module.exports = router; 