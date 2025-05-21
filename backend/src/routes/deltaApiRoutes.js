/**
 * Delta Exchange API Routes
 * Defines routes for Delta Exchange API operations
 */

const express = require('express');
const deltaApiController = require('../controllers/deltaApiController');
const { authenticate } = require('../middleware/auth');

const router = express.Router();

// Public endpoints (no authentication required)
// Get all markets
router.get('/markets', deltaApiController.getMarkets);

// Authentication required for the following routes
router.use(authenticate);

// Market data
router.get('/markets/:symbol', deltaApiController.getMarketData);

// Order management
router.post('/orders', deltaApiController.placeOrder);
router.get('/orders', deltaApiController.getActiveOrders);
router.delete('/orders/:orderId', deltaApiController.cancelOrder);

// Account and wallet
router.get('/wallet/balances', deltaApiController.getWalletBalances);
router.get('/positions', deltaApiController.getPositions);

module.exports = router; 