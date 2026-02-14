import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ChartSpec } from '../api/client';

export function formatValue(value: number, format: string): string {
  if (format === 'currency') {
    if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
    if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
    return `$${value.toLocaleString()}`;
  }
  if (format === 'percent') return `${value}%`;
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return value.toLocaleString();
}

export function formatTooltipValue(value: number, format: string): string {
  if (format === 'currency') return `$${value.toLocaleString()}`;
  if (format === 'percent') return `${value}%`;
  return value.toLocaleString();
}

export default function ChartRenderer({ spec }: { spec: ChartSpec }) {
  if (!spec || !Array.isArray(spec.data) || spec.data.length === 0) {
    return null;
  }

  const getColor = (entry: Record<string, any>, index: number): string => {
    if (spec.color_key && typeof spec.colors === 'object' && spec.color_key in (entry || {})) {
      return (spec.colors as Record<string, string>)[entry[spec.color_key]] || '#6366f1';
    }
    if (typeof spec.colors === 'string') return spec.colors;
    return '#6366f1';
  };

  const yFmt = spec.y_format || 'number';

  return (
    <div style={{
      flex: '1 1 340px',
      background: 'var(--bg-tertiary)',
      borderRadius: '12px',
      padding: '20px',
      minWidth: '300px',
    }}>
      <h4 style={{ margin: '0 0 16px', fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)' }}>
        {spec.title}
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        {spec.type === 'line' ? (
          <LineChart data={spec.data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey={spec.x_key} tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatValue(v, yFmt)} />
            <Tooltip
              formatter={(value) => [formatTooltipValue(Number(value), yFmt), spec.y_label || spec.y_key]}
              contentStyle={{ fontSize: '12px', borderRadius: '8px' }}
            />
            <Line
              type="monotone"
              dataKey={spec.y_key}
              stroke={typeof spec.colors === 'string' ? spec.colors : '#6366f1'}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        ) : spec.type === 'horizontal_bar' ? (
          <BarChart data={spec.data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatValue(v, yFmt)} />
            <YAxis type="category" dataKey={spec.x_key} tick={{ fontSize: 12 }} width={90} />
            <Tooltip
              formatter={(value) => [formatTooltipValue(Number(value), yFmt), spec.y_label || spec.y_key]}
              contentStyle={{ fontSize: '12px', borderRadius: '8px' }}
            />
            <Bar dataKey={spec.y_key} radius={[0, 4, 4, 0]}>
              {spec.data.map((entry, i) => (
                <Cell key={i} fill={getColor(entry, i)} />
              ))}
            </Bar>
          </BarChart>
        ) : (
          <BarChart data={spec.data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis dataKey={spec.x_key} tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatValue(v, yFmt)} />
            <Tooltip
              formatter={(value) => [formatTooltipValue(Number(value), yFmt), spec.y_label || spec.y_key]}
              contentStyle={{ fontSize: '12px', borderRadius: '8px' }}
            />
            <Bar dataKey={spec.y_key} radius={[4, 4, 0, 0]}>
              {spec.data.map((entry, i) => (
                <Cell key={i} fill={getColor(entry, i)} />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
