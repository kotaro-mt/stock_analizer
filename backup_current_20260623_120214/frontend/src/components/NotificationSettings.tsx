import { useState } from 'react';
import { Bell, BellOff } from 'lucide-react';

interface NotificationSettingsProps {
  ticker: string;
  info: any;
  onUpdate: () => void;
}

export default function NotificationSettings({ ticker, info, onUpdate }: NotificationSettingsProps) {
  const [updating, setUpdating] = useState(false);
  const enabled = info?.notifications_enabled ?? true;

  const toggleNotifications = async () => {
    try {
      setUpdating(true);
      await fetch('/api/config/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, enabled: !enabled })
      });
      onUpdate();
    } catch (err) {
      console.error("Failed to update", err);
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto' }}>
      <div style={{
        background: 'rgba(30,41,59,0.6)', borderRadius: '16px',
        border: '1px solid rgba(255,255,255,0.08)', padding: '32px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
          <div>
            <h2 style={{ margin: '0 0 6px', fontSize: '20px', fontWeight: 700, color: '#fff' }}>Alert Settings</h2>
            <p style={{ margin: 0, color: '#64748B', fontSize: '13px' }}>
              Manage notifications for <strong style={{ color: '#CBD5E1' }}>{ticker}</strong>
            </p>
          </div>
          <button onClick={toggleNotifications} disabled={updating} style={{
            display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
            borderRadius: '10px', fontWeight: 600, fontSize: '13px', cursor: 'pointer', border: 'none',
            background: enabled ? 'rgba(52,211,153,0.15)' : 'rgba(30,41,59,0.8)',
            color: enabled ? '#34D399' : '#64748B',
            outline: enabled ? '1px solid rgba(52,211,153,0.3)' : '1px solid rgba(255,255,255,0.06)',
            transition: 'all 0.2s', opacity: updating ? 0.6 : 1
          }}>
            {enabled ? <Bell size={16} /> : <BellOff size={16} />}
            {enabled ? '通知 ON' : '通知 OFF'}
          </button>
        </div>

        {!enabled && (
          <div style={{
            padding: '14px 16px', borderRadius: '10px',
            background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.2)',
            color: '#7DD3FC', fontSize: '13px', marginBottom: '24px'
          }}>
            ℹ この銘柄の通知はOFFです。設定は保持されますが、アラートは発火しません。
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {['週足 MACD', '日足 MACD', '価格アラート', 'トレンドライン'].map(label => (
            <div key={label} style={{
              padding: '16px', borderRadius: '10px',
              background: 'rgba(15,23,42,0.5)', border: '1px solid rgba(255,255,255,0.05)',
              opacity: enabled ? 1 : 0.4
            }}>
              <div style={{ fontWeight: 600, fontSize: '13px', color: '#CBD5E1', marginBottom: '4px' }}>{label}</div>
              <div style={{ fontSize: '11px', color: '#475569' }}>グローバル設定を使用</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
