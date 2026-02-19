import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { getUserInfo, saveCurrentUser } from '../api/client';

export default function AuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('Completing authentication...');

  useEffect(() => {
    const handleCallback = async () => {
      const userId = searchParams.get('user_id');
      const email = searchParams.get('email');
      const error = searchParams.get('error');

      if (error) {
        setStatus('error');
        setMessage(`Authentication failed: ${error}`);
        return;
      }

      if (!userId || !email) {
        setStatus('error');
        setMessage('Invalid callback parameters');
        return;
      }

      try {
        // Fetch full user info
        const userInfo = await getUserInfo(userId);
        saveCurrentUser(userInfo);

        setStatus('success');
        setMessage(`Welcome, ${userInfo.name || userInfo.email}!`);

        // Redirect to home (data rooms) after short delay.
        // Google-authenticated users can navigate to /drive from there.
        setTimeout(() => {
          navigate('/');
        }, 1500);
      } catch (err: any) {
        setStatus('error');
        setMessage(err.response?.data?.detail || 'Failed to complete authentication');
      }
    };

    handleCallback();
  }, [searchParams, navigate]);

  return (
    <div className="auth-callback">
      <div className="callback-card">
        {status === 'loading' && (
          <>
            <Loader className="icon spinning" />
            <h2>Authenticating</h2>
            <p>{message}</p>
          </>
        )}

        {status === 'success' && (
          <>
            <CheckCircle className="icon success" />
            <h2>Success!</h2>
            <p>{message}</p>
            <p className="redirect-text">Redirecting to your data rooms...</p>
          </>
        )}

        {status === 'error' && (
          <>
            <AlertCircle className="icon error" />
            <h2>Authentication Failed</h2>
            <p>{message}</p>
            <button className="btn btn-primary" onClick={() => navigate('/')}>
              Return Home
            </button>
          </>
        )}
      </div>

      <style>{`
        .auth-callback {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 80vh;
          padding: 24px;
        }

        .callback-card {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 48px;
          text-align: center;
          max-width: 400px;
          width: 100%;
        }

        .callback-card .icon {
          width: 64px;
          height: 64px;
          margin-bottom: 24px;
        }

        .callback-card .icon.success {
          color: #34a853;
        }

        .callback-card .icon.error {
          color: #ea4335;
        }

        .callback-card h2 {
          margin: 0 0 12px 0;
          font-size: 24px;
        }

        .callback-card p {
          margin: 0 0 8px 0;
          color: var(--text-secondary);
        }

        .redirect-text {
          font-size: 14px;
          margin-top: 16px !important;
        }

        .spinning {
          animation: spin 1s linear infinite;
          color: var(--primary-color, #4285f4);
        }

        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 12px 24px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          margin-top: 24px;
        }

        .btn-primary {
          background: var(--primary-color, #4285f4);
          color: white;
        }

        .btn-primary:hover {
          background: var(--primary-hover, #3367d6);
        }
      `}</style>
    </div>
  );
}
