import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  DeepPartial,
  ChartOptions,
  LineData,
  ColorType,
} from 'lightweight-charts';
import { io, Socket } from 'socket.io-client';

interface TVChartContainerProps {
  symbol: string;
  interval: string;
  data: Array<{ time: number | string; value: number }>;
  signals?: Array<{ time: number | string; type: 'buy' | 'sell' | 'hold'; confidence?: number }>;
}

export const TVChartContainer: React.FC<TVChartContainerProps> = ({
  symbol,
  interval,
  data: initialData,
  signals,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [chartData, setChartData] = useState(initialData);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Connect to backend WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3001';
    const socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      timeout: 10000,
    });
    socketRef.current = socket;

    socket.on('connect', () => {
      socket.emit('subscribe:market', symbol);
    });
    socket.on('market:data', (payload) => {
      if (payload.symbol === symbol && payload.data) {
        // Assume payload.data is array of { time, value }
        setChartData((prev) => {
          // Merge or replace logic as needed
          if (Array.isArray(payload.data)) return payload.data;
          if (payload.data.time && payload.data.value) return [...prev, payload.data];
          return prev;
        });
      }
    });
    socket.on('disconnect', () => {
      // Optionally handle disconnect
    });
    socket.on('error', (err) => {
      // Optionally handle error
      // eslint-disable-next-line no-console
      console.error('WebSocket error:', err);
    });
    return () => {
      socket.emit('unsubscribe:market', symbol);
      socket.disconnect();
    };
  }, [symbol]);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    if (chartRef.current) {
      chartRef.current.remove();
    }
    const chartOptions: DeepPartial<ChartOptions> = {
      layout: {
        background: { type: ColorType.Solid, color: '#181A20' },
        textColor: '#D9D9D9',
      },
      grid: {
        vertLines: { color: '#222' },
        horzLines: { color: '#222' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      timeScale: { timeVisible: true, secondsVisible: false },
    };
    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;
    // @ts-expect-error: addLineSeries is available at runtime in lightweight-charts v5
    const lineSeries = chart.addLineSeries({ color: '#2962FF', lineWidth: 2 });
    lineSeries.setData(chartData as LineData[]);
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current!.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [symbol, interval, chartData]);

  return <div style={{ width: '100%', height: 500 }} ref={chartContainerRef} />;
};
