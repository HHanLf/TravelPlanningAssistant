import { motion } from 'framer-motion'
import {
  Bookmark,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  Compass,
  History,
  MapPinned,
  Plus,
  Settings,
  Sparkles,
} from 'lucide-react'

const NAV_ITEMS = [
  { label: '新建会话', icon: Plus, active: true },
  { label: '历史记录', icon: History },
  { label: '收藏路线', icon: Bookmark },
  { label: '我的行程', icon: CalendarDays },
  { label: '设置', icon: Settings },
]

export function Sidebar({ collapsed, onToggle, sessionId, profile = {}, stats = {} }) {
  const profileItems = [
    ['目的地', profile.destination],
    ['出发地', profile.departure || profile.origin],
    ['天数', profile.days ? `${profile.days} 天` : ''],
    ['预算', profile.budget ? `¥${profile.budget}` : ''],
    ['人数', profile.companions || profile.group_size ? `${profile.companions || profile.group_size} 人` : ''],
  ].filter(([, value]) => value)

  return (
    <motion.aside
      className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}
      initial={{ x: -24, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      <div className="sidebar__brand">
        <div className="brand-icon">
          <Compass size={22} />
        </div>
        {!collapsed && (
          <div>
            <p>AI Travel</p>
            <strong>Planner</strong>
          </div>
        )}
        <button type="button" className="icon-button sidebar__toggle" onClick={onToggle} aria-label="折叠侧边栏">
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="sidebar__nav" aria-label="主导航">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <button key={item.label} type="button" className={`nav-item ${item.active ? 'nav-item--active' : ''}`}>
              <Icon size={18} />
              {!collapsed && <span>{item.label}</span>}
            </button>
          )
        })}
      </nav>

      {!collapsed && (
        <>
          <section className="side-card side-card--accent">
            <div className="side-card__title">
              <Sparkles size={16} />
              <span>当前会话</span>
            </div>
            <strong>{sessionId}</strong>
            <div className="mini-grid">
              <div>
                <span>消息</span>
                <b>{stats.messageCount || 0}</b>
              </div>
              <div>
                <span>工具</span>
                <b>{stats.toolCount || 0}</b>
              </div>
            </div>
          </section>

          <section className="side-card">
            <div className="side-card__title">
              <MapPinned size={16} />
              <span>旅行画像</span>
            </div>
            {profileItems.length ? (
              <div className="profile-list">
                {profileItems.map(([label, value]) => (
                  <div className="profile-row" key={label}>
                    <span>{label}</span>
                    <b>{value}</b>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">告诉我目的地、预算和偏好后，这里会自动沉淀旅行画像。</p>
            )}
          </section>
        </>
      )}
    </motion.aside>
  )
}

