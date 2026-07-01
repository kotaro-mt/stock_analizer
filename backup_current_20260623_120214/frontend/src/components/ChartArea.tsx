import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Loader2, AlertTriangle } from 'lucide-react';

export default function ChartArea({ ticker }: { ticker: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`/api/chart/${ticker}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        setData(result);
      } catch (err: any) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    };
    fetchChartData();
  }, [ticker]);

  if (loading) {
    return (
      <div style={{
        height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: '#64748B',
        background: 'rgba(30,41,59,0.4)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.06)'
      }}>
        <div style={{ color: '#38BDF8', marginBottom: '12px', animation: 'spin 1s linear infinite' }}>
          <Loader2 size={36} />
        </div>
        <p style={{ margin: 0, fontSize: '14px' }}>Loading market data...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{
        height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: '#F87171',
        background: 'rgba(30,41,59,0.4)', borderRadius: '16px', border: '1px solid rgba(248,113,113,0.2)'
      }}>
        <AlertTriangle size={36} style={{ marginBottom: '12px' }} />
        <p style={{ margin: 0 }}>データの読み込みに失敗しました</p>
        <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#94A3B8' }}>{error}</p>
      </div>
    );
  }

  const ohlcv = data.data;
  const dates = ohlcv.map((d: any) => d.Date);
  const opens = ohlcv.map((d: any) => d.Open);
  const highs = ohlcv.map((d: any) => d.High);
  const lows = ohlcv.map((d: any) => d.Low);
  const closes = ohlcv.map((d: any) => d.Close);
  const volumes = ohlcv.map((d: any) => d.Volume);
  const macd = ohlcv.map((d: any) => d.macd);
  const macd_signal = ohlcv.map((d: any) => d.macd_signal);
  const macd_hist = ohlcv.map((d: any) => d.macd_hist);
  const rsi = ohlcv.map((d: any) => d.rsi);

  return (
    <div style={{
      height: '100%', background: 'rgba(30,41,59,0.4)', borderRadius: '16px',
      border: '1px solid rgba(255,255,255,0.06)', overflow: 'hidden', display: 'flex', flexDirection: 'column'
    }}>
      <Plot
        data={[
          {
            x: dates, open: opens, high: highs, low: lows, close: closes,
            type: 'candlestick', name: 'Price',
            increasing: { line: { color: '#34D399' }, fillcolor: '#34D399' },
            decreasing: { line: { color: '#F87171' }, fillcolor: '#F87171' },
            yaxis: 'y',
          },
          {
            x: dates, y: volumes, type: 'bar', name: 'Volume',
            marker: { color: 'rgba(148,163,184,0.25)' }, yaxis: 'y2',
          },
          {
            x: dates, y: macd, type: 'scatter', mode: 'lines', name: 'MACD',
            line: { color: '#38BDF8', width: 1.5 }, yaxis: 'y3',
          },
          {
            x: dates, y: macd_signal, type: 'scatter', mode: 'lines', name: 'Signal',
            line: { color: '#F472B6', width: 1.5 }, yaxis: 'y3',
          },
          {
            x: dates, y: macd_hist, type: 'bar', name: 'Hist',
            marker: { color: '#818CF8', opacity: 0.6 }, yaxis: 'y3',
          },
          {
            x: dates, y: rsi, type: 'scatter', mode: 'lines', name: 'RSI',
            line: { color: '#A78BFA', width: 1.5 }, yaxis: 'y4',
          },
        ]}
        layout={{
          autosize: true,
          margin: { t: 20, l: 60, r: 60, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#94A3B8', family: 'Inter, sans-serif', size: 11 },
          xaxis: { rangeslider: { visible: false }, gridcolor: 'rgba(255,255,255,0.04)', domain: [0, 1] },
          yaxis: { domain: [0.4, 1], gridcolor: 'rgba(255,255,255,0.04)', tickformat: ',.0f', side: 'right' },
          yaxis2: { domain: [0.4, 1], overlaying: 'y', side: 'left', showgrid: false, showticklabels: false },
          yaxis3: { domain: [0.2, 0.38], gridcolor: 'rgba(255,255,255,0.04)', side: 'right' },
          yaxis4: { domain: [0, 0.18], gridcolor: 'rgba(255,255,255,0.04)', range: [0, 100], side: 'right' },
          showlegend: false, dragmode: 'pan',
        } as any}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
}
