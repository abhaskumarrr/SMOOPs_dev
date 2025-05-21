const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const EventEmitter = require('events');
const analyticsEmitter = new EventEmitter();

function calculateSharpe(returns, riskFreeRate = 0) {
  if (!returns.length) return 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / returns.length);
  return std ? (mean - riskFreeRate) / std : 0;
}

async function recordMetric(name, value, tags = {}) {
  await prisma.metric.create({
    data: {
      name,
      value,
      recordedAt: new Date(),
      tags: JSON.stringify(tags),
    },
  });
  analyticsEmitter.emit('metric', { name, value, tags });
}

async function onTradeExecuted(trade) {
  // Example: Compute PnL
  const pnl = trade.exitPrice - trade.entryPrice;
  await recordMetric('PnL', pnl, { userId: trade.userId, symbol: trade.symbol });
  // Compute win rate, Sharpe, drawdown, etc. (implement as needed)
  // ...
}

module.exports = { onTradeExecuted, recordMetric, analyticsEmitter }; 