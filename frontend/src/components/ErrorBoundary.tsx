import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fullPage?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] Caught render error:', error);
    console.error('[ErrorBoundary] Component stack:', errorInfo.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="empty-state" style={{ height: this.props.fullPage ? '60vh' : 'auto', padding: '48px 24px' }}>
          <div className="empty-icon" style={{ background: 'var(--error-soft)' }}>
            <AlertCircle size={32} style={{ color: 'var(--error)' }} />
          </div>
          <h3 className="empty-title">Something went wrong</h3>
          <p className="empty-description">
            An unexpected error occurred. Please try reloading.
          </p>
          {import.meta.env.DEV && this.state.error && (
            <pre style={{
              fontSize: '12px',
              color: 'var(--text-tertiary)',
              background: 'var(--bg-tertiary)',
              padding: '12px',
              borderRadius: '8px',
              maxWidth: '500px',
              overflow: 'auto',
              marginTop: '12px',
            }}>
              {this.state.error.message}
            </pre>
          )}
          <button
            className="btn btn-primary"
            onClick={this.handleReset}
            style={{ marginTop: '16px' }}
          >
            <RefreshCw size={16} />
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
