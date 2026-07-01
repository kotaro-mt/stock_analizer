import { BarChart3, Search, AlertTriangle } from 'lucide-react';

interface SidebarProps {
  tickers: any[];
  selected: string | null;
  onSelect: (ticker: string) => void;
  loading: boolean;
  apiError?: string | null;
}

export default function Sidebar({ tickers, selected, onSelect, loading, apiError }: SidebarProps) {
  return (
    <aside style={{
      width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column',
      background: 'rgba(15,23,42,0.9)', borderRight: '1px solid rgba(255,255,255,0.08)',
      backdropFilter: 'blur(12px)', zIndex: 20
    }}>
      {/* Logo */}
      <div style={{ padding: '24px 20px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, #38BDF8, #818CF8)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 700, color: '#fff', fontSize: '14px'
          }}>S</div>
          <span style={{ fontSize: '16px', fontWeight: 700, color: '#E2E8F0' }}>StockFuture</span>
        </div>

        <div style={{ position: 'relative' }}>
          <Search style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#475569' }} size={14} />
          <input placeholder="Search tickers..." style={{
            width: '100%', background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '8px', padding: '8px 8px 8px 32px', fontSize: '13px', color: '#CBD5E1',
            outline: 'none', boxSizing: 'border-box'
          }} />
        </div>
      </div>

      {/* Ticker List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '8px', paddingLeft: '8px' }}>
          Watchlist
        </div>

        {apiError ? (
          <div style={{ padding: '16px', color: '#F87171', fontSize: '12px', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
            <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: '1px' }} />
            <span>APIサーバーに接続できません</span>
          </div>
        ) : loading ? (
          [1, 2, 3, 4].map(i => (
            <div key={i} style={{ height: '56px', background: 'rgba(30,41,59,0.5)', borderRadius: '10px', marginBottom: '6px' }} />
          ))
        ) : (
          tickers.map(t => {
            const isSelected = selected === t.ticker;
            return (
              <button key={t.ticker} onClick={() => onSelect(t.ticker)} style={{
                width: '100%', display: 'flex', alignItems: 'center', gap: '12px',
                padding: '10px', borderRadius: '10px', marginBottom: '4px', cursor: 'pointer',
                background: isSelected ? 'rgba(56,189,248,0.1)' : 'transparent',
                border: isSelected ? '1px solid rgba(56,189,248,0.2)' : '1px solid transparent',
                textAlign: 'left', transition: 'all 0.15s'
              }}>
                <div style={{
                  width: '38px', height: '38px', borderRadius: '8px', flexShrink: 0,
                  background: isSelected ? 'rgba(56,189,248,0.15)' : 'rgba(30,41,59,0.8)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: isSelected ? '#38BDF8' : '#64748B'
                }}>
                  <BarChart3 size={17} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: isSelected ? '#fff' : '#CBD5E1' }}>
                    {t.ticker}
                  </div>
                  <div style={{ fontSize: '11px', color: '#475569', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.name}
                  </div>
                </div>
                {!t.notifications_enabled && (
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#475569', flexShrink: 0 }} />
                )}
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
