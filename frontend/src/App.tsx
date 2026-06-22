import { useState, useEffect } from 'react';
import { Activity, Bell, TrendingUp } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChartArea from './components/ChartArea';
import NotificationSettings from './components/NotificationSettings';

function App() {
  const [tickers, setTickers] = useState<any[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'chart' | 'settings'>('chart');
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchTickers();
  }, []);

  const fetchTickers = async () => {
    try {
      setLoading(true);
      setApiError(null);
      const res = await fetch('/api/tickers');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTickers(data);
      if (data.length > 0) {
        setSelectedTicker(data[0].ticker);
      }
    } catch (err: any) {
      console.error("Failed to fetch tickers", err);
      setApiError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const currentTickerInfo = tickers.find(t => t.ticker === selectedTicker);

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#0F172A', color: '#f1f5f9' }}>
      {/* Sidebar */}
      <Sidebar
        tickers={tickers}
        selected={selectedTicker}
        onSelect={setSelectedTicker}
        loading={loading}
        apiError={apiError}
      />

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

        {/* Header */}
        <header style={{
          height: '72px', padding: '0 32px', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(30,41,59,0.6)', backdropFilter: 'blur(12px)', flexShrink: 0
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '44px', height: '44px', borderRadius: '12px',
              background: 'rgba(56,189,248,0.15)', border: '1px solid rgba(56,189,248,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38BDF8'
            }}>
              <TrendingUp size={22} />
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: '#fff' }}>
                {currentTickerInfo ? currentTickerInfo.name : (loading ? 'Loading...' : 'StockFuture')}
                {currentTickerInfo && !currentTickerInfo.notifications_enabled && (
                  <span style={{
                    fontSize: '11px', fontWeight: 400, background: 'rgba(100,116,139,0.3)',
                    color: '#94A3B8', padding: '2px 8px', borderRadius: '20px',
                    border: '1px solid rgba(255,255,255,0.1)', marginLeft: '10px'
                  }}>通知OFF</span>
                )}
              </h1>
              <p style={{ margin: 0, fontSize: '13px', color: '#64748B' }}>{selectedTicker || ''}</p>
            </div>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            background: 'rgba(30,41,59,0.8)', padding: '4px', borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.08)'
          }}>
            {(['chart', 'settings'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '8px 16px', borderRadius: '8px', fontSize: '13px', fontWeight: 600,
                cursor: 'pointer', border: 'none', transition: 'all 0.2s',
                background: activeTab === tab
                  ? (tab === 'chart' ? '#38BDF8' : '#818CF8')
                  : 'transparent',
                color: activeTab === tab ? '#0F172A' : '#94A3B8',
              }}>
                {tab === 'chart' ? <Activity size={15} /> : <Bell size={15} />}
                {tab === 'chart' ? 'Chart' : 'Alerts'}
              </button>
            ))}
          </div>
        </header>

        {/* Content Area */}
        <main style={{ flex: 1, overflow: 'auto', padding: '24px', position: 'relative' }}>
          {apiError ? (
            <div style={{
              background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.3)',
              borderRadius: '12px', padding: '24px', color: '#FCA5A5'
            }}>
              <strong>⚠ APIサーバーに接続できません</strong>
              <p style={{ marginTop: '8px', fontSize: '13px', color: '#94A3B8' }}>
                バックエンドが起動していることを確認してください。<br />
                エラー: {apiError}
              </p>
            </div>
          ) : activeTab === 'chart' && selectedTicker ? (
            <ChartArea ticker={selectedTicker} />
          ) : activeTab === 'settings' && selectedTicker ? (
            <NotificationSettings ticker={selectedTicker} info={currentTickerInfo} onUpdate={fetchTickers} />
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;
