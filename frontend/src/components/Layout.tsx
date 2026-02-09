import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ReactNode, useState, useEffect } from 'react';
import { Cloud, LogIn, LogOut, User, ChevronDown } from 'lucide-react';
import {
  getCurrentUser,
  clearCurrentUser,
  initiateGoogleLogin,
  logoutUser,
  UserInfo,
} from '../api/client';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState<UserInfo | null>(null);
  const [showUserMenu, setShowUserMenu] = useState(false);

  useEffect(() => {
    const storedUser = getCurrentUser();
    setUser(storedUser);
  }, [location]);

  const handleLogin = async () => {
    try {
      const { auth_url } = await initiateGoogleLogin();
      window.location.href = auth_url;
    } catch (err) {
      console.error('Failed to initiate login:', err);
    }
  };

  const handleLogout = async () => {
    if (user) {
      try {
        await logoutUser(user.id);
      } catch (err) {
        console.error('Logout error:', err);
      }
    }
    clearCurrentUser();
    setUser(null);
    setShowUserMenu(false);
    navigate('/');
  };

  const isActive = (path: string) => {
    if (path === '/drive') {
      return location.pathname.startsWith('/drive');
    }
    return location.pathname === path;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          background: 'var(--glass-bg-solid)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: '1px solid var(--border-subtle)',
          padding: '12px 0',
        }}
      >
        <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '40px' }}>
            {/* Logo */}
            <Link
              to="/"
              style={{
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}
            >
              {/* New Gradient Logo Icon */}
              <div
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  background: 'var(--gradient-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 24px var(--accent-glow)',
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M12 2C9.5 2 7.5 3.5 7 5.5C5.5 5.8 4.3 7 4 8.5C3.5 10.5 4.5 12.5 6.5 13.5C6 14.5 6 15.5 6.5 16.5C7 18 8.5 19 10 19H14C15.5 19 17 18 17.5 16.5C18 15.5 18 14.5 17.5 13.5C19.5 12.5 20.5 10.5 20 8.5C19.7 7 18.5 5.8 17 5.5C16.5 3.5 14.5 2 12 2Z"
                    fill="white"
                    opacity="0.9"
                  />
                  <path
                    d="M9 11C9.55228 11 10 10.5523 10 10C10 9.44772 9.55228 9 9 9C8.44772 9 8 9.44772 8 10C8 10.5523 8.44772 11 9 11Z"
                    fill="rgba(139, 92, 246, 0.8)"
                  />
                  <path
                    d="M15 11C15.5523 11 16 10.5523 16 10C16 9.44772 15.5523 9 15 9C14.4477 9 14 9.44772 14 10C14 10.5523 14.4477 11 15 11Z"
                    fill="rgba(139, 92, 246, 0.8)"
                  />
                  <path
                    d="M12 16C13.6569 16 15 14.8807 15 14H9C9 14.8807 10.3431 16 12 16Z"
                    fill="rgba(139, 92, 246, 0.8)"
                  />
                </svg>
              </div>
              <span
                style={{
                  fontSize: '18px',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  letterSpacing: '-0.02em',
                }}
              >
                Noesis AI
              </span>
            </Link>

            {/* Navigation */}
            <nav style={{ display: 'flex', gap: '6px' }}>
              <Link
                to="/"
                style={{
                  textDecoration: 'none',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 500,
                  color: isActive('/') ? 'var(--accent-tertiary)' : 'var(--text-secondary)',
                  background: isActive('/') ? 'var(--accent-glow)' : 'transparent',
                  transition: 'all 0.2s ease',
                }}
              >
                Data Rooms
              </Link>
              <Link
                to="/upload"
                style={{
                  textDecoration: 'none',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: 500,
                  color: isActive('/upload') ? 'var(--accent-tertiary)' : 'var(--text-secondary)',
                  background: isActive('/upload') ? 'var(--accent-glow)' : 'transparent',
                  transition: 'all 0.2s ease',
                }}
              >
                Upload
              </Link>
              {user && (
                <Link
                  to="/drive"
                  style={{
                    textDecoration: 'none',
                    padding: '8px 16px',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: 500,
                    color: isActive('/drive') ? 'var(--accent-tertiary)' : 'var(--text-secondary)',
                    background: isActive('/drive') ? 'var(--accent-glow)' : 'transparent',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <Cloud style={{ width: '16px', height: '16px' }} />
                  Google Drive
                </Link>
              )}
            </nav>
          </div>

          {/* User Menu */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {user ? (
              <div style={{ position: 'relative' }}>
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '6px 12px 6px 6px',
                    border: '1px solid var(--border-default)',
                    borderRadius: '10px',
                    background: 'var(--bg-tertiary)',
                    cursor: 'pointer',
                    fontSize: '14px',
                    color: 'var(--text-primary)',
                    transition: 'all 0.2s ease',
                  }}
                >
                  {user.picture_url ? (
                    <img
                      src={user.picture_url}
                      alt={user.name || user.email}
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '6px',
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '6px',
                        background: 'var(--gradient-primary)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <User style={{ width: '16px', height: '16px', color: 'white' }} />
                    </div>
                  )}
                  <span
                    style={{
                      maxWidth: '120px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontWeight: 500,
                    }}
                  >
                    {user.name || user.email}
                  </span>
                  <ChevronDown
                    style={{
                      width: '14px',
                      height: '14px',
                      color: 'var(--text-tertiary)',
                      transform: showUserMenu ? 'rotate(180deg)' : 'rotate(0)',
                      transition: 'transform 0.2s ease',
                    }}
                  />
                </button>

                {showUserMenu && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '100%',
                      right: 0,
                      marginTop: '8px',
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border-default)',
                      borderRadius: '12px',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px var(--accent-glow)',
                      minWidth: '220px',
                      zIndex: 100,
                      overflow: 'hidden',
                      animation: 'slideInDown 0.2s ease',
                    }}
                  >
                    <div
                      style={{
                        padding: '14px 16px',
                        borderBottom: '1px solid var(--border-subtle)',
                      }}
                    >
                      <div style={{ fontWeight: 500, fontSize: '14px', color: 'var(--text-primary)' }}>
                        {user.name}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                        {user.email}
                      </div>
                    </div>
                    <button
                      onClick={handleLogout}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        width: '100%',
                        padding: '12px 16px',
                        border: 'none',
                        background: 'transparent',
                        cursor: 'pointer',
                        fontSize: '14px',
                        color: 'var(--error)',
                        textAlign: 'left',
                        transition: 'background 0.15s ease',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'var(--error-soft)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      <LogOut style={{ width: '16px', height: '16px' }} />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <button
                onClick={handleLogin}
                className="btn btn-primary"
                style={{
                  padding: '10px 18px',
                }}
              >
                <LogIn style={{ width: '16px', height: '16px' }} />
                Connect Google Drive
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, padding: '32px 0' }}>
        <div className="container">{children}</div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '16px 0',
          marginTop: 'auto',
        }}
      >
        <div
          className="container"
          style={{
            textAlign: 'center',
            fontSize: '12px',
            color: 'var(--text-tertiary)',
          }}
        >
          Noesis AI - Intelligent Due Diligence
        </div>
      </footer>

      {/* Click outside to close menu */}
      {showUserMenu && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 99,
          }}
          onClick={() => setShowUserMenu(false)}
        />
      )}
    </div>
  );
}
