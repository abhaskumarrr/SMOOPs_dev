import dynamic from 'next/dynamic';
import React from 'react';

const TVChartContainer = dynamic(
  () => import('../components/charts/TVChartContainer').then((mod) => mod.TVChartContainer),
  { ssr: false },
);

const mockData = [
  { time: 1642425322, value: 42000 },
  { time: 1642511722, value: 42500 },
  { time: 1642598122, value: 43000 },
  { time: 1642684522, value: 42800 },
  { time: 1642770922, value: 43200 },
  { time: 1642857322, value: 43500 },
  { time: 1642943722, value: 44000 },
  { time: 1643030122, value: 43800 },
  { time: 1643116522, value: 44500 },
  { time: 1643202922, value: 45000 },
];

export default function Dashboard() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#181A20' }}>
      <aside style={{ width: 220, background: '#20232a', color: '#fff', padding: 24 }}>
        <h2>SMOOPs Dashboard</h2>
        <nav>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li>Chart</li>
            <li>Signals</li>
            <li>Settings</li>
          </ul>
        </nav>
      </aside>
      <main style={{ flex: 1, padding: 32 }}>
        <h1 style={{ color: '#fff' }}>BTCUSD Trading Chart</h1>
        <TVChartContainer symbol="BTCUSD" interval="1h" data={mockData} />
      </main>
    </div>
  );
}
