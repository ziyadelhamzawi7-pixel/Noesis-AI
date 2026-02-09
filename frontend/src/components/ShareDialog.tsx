import { useState, useEffect } from 'react';
import { X, UserPlus, Trash2, Users, Mail, Crown, Clock } from 'lucide-react';
import {
  getDataRoomMembers,
  inviteDataRoomMember,
  removeDataRoomMember,
  DataRoomMember,
} from '../api/client';

interface ShareDialogProps {
  dataRoomId: string;
  companyName: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function ShareDialog({ dataRoomId, companyName, isOpen, onClose }: ShareDialogProps) {
  const [members, setMembers] = useState<DataRoomMember[]>([]);
  const [inviteEmail, setInviteEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInviting, setIsInviting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadMembers();
    }
  }, [isOpen, dataRoomId]);

  const loadMembers = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getDataRoomMembers(dataRoomId);
      setMembers(result.members);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load members');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;

    setIsInviting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const result = await inviteDataRoomMember(dataRoomId, inviteEmail.trim());
      setSuccessMessage(result.message);
      setInviteEmail('');
      await loadMembers();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to invite member');
    } finally {
      setIsInviting(false);
    }
  };

  const handleRemove = async (memberId: string, email: string) => {
    if (!confirm(`Remove ${email} from this data room?`)) return;

    try {
      await removeDataRoomMember(dataRoomId, memberId);
      setMembers(members.filter((m) => m.id !== memberId));
      setSuccessMessage(`Removed ${email}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to remove member');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="share-dialog-overlay" onClick={onClose}>
      <div className="share-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="share-dialog-header">
          <div className="share-dialog-title">
            <Users size={20} />
            <span>Share "{companyName}"</span>
          </div>
          <button className="share-dialog-close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <form className="share-invite-form" onSubmit={handleInvite}>
          <div className="share-invite-input-row">
            <Mail size={16} className="share-invite-icon" />
            <input
              type="email"
              placeholder="Enter email to invite..."
              value={inviteEmail}
              onChange={(e) => setInviteEmail(e.target.value)}
              disabled={isInviting}
              className="share-invite-input"
            />
            <button
              type="submit"
              disabled={isInviting || !inviteEmail.trim()}
              className="share-invite-button"
            >
              <UserPlus size={16} />
              {isInviting ? 'Inviting...' : 'Invite'}
            </button>
          </div>
        </form>

        {error && <div className="share-message share-error">{error}</div>}
        {successMessage && <div className="share-message share-success">{successMessage}</div>}

        <div className="share-members-list">
          <div className="share-members-header">
            Members ({members.length})
          </div>

          {isLoading ? (
            <div className="share-loading">Loading members...</div>
          ) : (
            members.map((member) => (
              <div key={member.id} className="share-member-item">
                <div className="share-member-info">
                  {member.picture_url ? (
                    <img
                      src={member.picture_url}
                      alt={member.name || member.invited_email}
                      className="share-member-avatar"
                    />
                  ) : (
                    <div className="share-member-avatar share-member-avatar-placeholder">
                      {(member.name || member.invited_email)[0].toUpperCase()}
                    </div>
                  )}
                  <div className="share-member-details">
                    <span className="share-member-name">
                      {member.name || member.invited_email}
                    </span>
                    {member.name && (
                      <span className="share-member-email">{member.invited_email}</span>
                    )}
                  </div>
                </div>
                <div className="share-member-actions">
                  {member.role === 'owner' ? (
                    <span className="share-role-badge share-role-owner">
                      <Crown size={12} /> Owner
                    </span>
                  ) : (
                    <>
                      {member.status === 'pending' ? (
                        <span className="share-status-badge share-status-pending">
                          <Clock size={12} /> Pending
                        </span>
                      ) : (
                        <span className="share-role-badge share-role-member">Member</span>
                      )}
                      <button
                        className="share-remove-button"
                        onClick={() => handleRemove(member.id, member.invited_email)}
                        title="Remove member"
                      >
                        <Trash2 size={14} />
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
