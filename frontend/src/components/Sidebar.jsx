import {
  ChevronLeft,
  ChevronRight,
  Compass,
  History,
  Library,
  Menu,
  MessageSquarePlus,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

const RECENT_PLANS = [
  { title: '东京 5 天游学路线', meta: '预算 8000 · 家庭出行' },
  { title: '云南 4 天慢游', meta: '预算 5000 · 情侣/朋友' },
  { title: '欧洲 10 天首访', meta: '签证/交通待整理' },
]

const NAV_ITEMS = [
  { label: '新建规划', icon: MessageSquarePlus, active: true },
  { label: '搜索规划', icon: Search },
  { label: '历史记录', icon: History },
  { label: '路线库', icon: Library },
  { label: '设置', icon: Settings2 },
]

export function Sidebar({
  collapsed,
  mobileOpen,
  onToggleCollapse,
  onCloseMobile,
  onNewChat,
  sessionId,
  profile = {},
  stats = {},
  apiStatus = {},
}) {
  const profileItems = [
    ['目的地', profile.destination],
    ['出发地', profile.departure || profile.origin],
    ['天数', profile.days ? `${profile.days} 天` : ''],
    ['预算', profile.budget ? `¥${profile.budget}` : ''],
    ['人数', profile.companions || profile.group_size ? `${profile.companions || profile.group_size} 人` : ''],
  ].filter(([, value]) => value)

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''} ${mobileOpen ? 'sidebar--mobile-open' : ''}`}>
      <div className="sidebar__top">
        <div className="sidebar__brand">
          <div className="brand-icon">
            <Compass size={18} />
          </div>
          {!collapsed ? (
            <div className="sidebar__brand-copy">
              <strong>旅行规划助手</strong>
              <p>Travel Planning Assistant</p>
            </div>
          ) : null}
        </div>

        <div className="sidebar__actions">
          <button type="button" className="icon-button" onClick={onNewChat} aria-label="新建规划">
            <MessageSquarePlus size={18} />
          </button>
          <button
            type="button"
            className="icon-button sidebar__toggle"
            onClick={onToggleCollapse}
            aria-label={collapsed ? '展开侧边栏' : '折叠侧边栏'}
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
          <button type="button" className="icon-button sidebar__mobile-close" onClick={onCloseMobile} aria-label="关闭侧边栏">
            <Menu size={18} />
          </button>
        </div>
      </div>

      {!collapsed ? (
        <button type="button" className="new-chat-button" onClick={onNewChat}>
          <Sparkles size={16} />
          <span>新建规划</span>
        </button>
      ) : null}

      <nav className="sidebar__nav" aria-label="侧边导航">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <button key={item.label} type="button" className={`nav-item ${item.active ? 'nav-item--active' : ''}`}>
              <Icon size={18} />
              {!collapsed ? <span>{item.label}</span> : null}
            </button>
          )
        })}
      </nav>

      {!collapsed ? (
        <>
          <section className="sidebar-card sidebar-card--compact">
            <div className="sidebar-card__label">
              <ShieldCheck size={15} />
              <span>连接状态</span>
            </div>
            <div className={`connection-row connection-row--${apiStatus.state || 'checking'}`}>
              <span />
              <p>{apiStatus.message || '正在连接后端'}</p>
            </div>
          </section>

          <section className="sidebar-card">
            <div className="sidebar-card__label">
              <History size={15} />
              <span>最近规划</span>
            </div>
            <div className="recent-list">
              {RECENT_PLANS.map((item, index) => (
                <button type="button" className={`recent-item ${index === 0 ? 'recent-item--active' : ''}`} key={item.title}>
                  <strong>{item.title}</strong>
                  <span>{item.meta}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="sidebar-card sidebar-card--compact">
            <div className="sidebar-card__label">
              <MessageSquarePlus size={15} />
              <span>当前会话</span>
            </div>
            <strong>{sessionId}</strong>
            <div className="sidebar-stats">
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

          {profileItems.length ? (
            <section className="sidebar-card">
              <div className="sidebar-card__label">
                <ShieldCheck size={15} />
                <span>旅程画像</span>
              </div>
              <div className="profile-list">
                {profileItems.map(([label, value]) => (
                  <div className="profile-row" key={label}>
                    <span>{label}</span>
                    <b>{value}</b>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
        </>
      ) : null}

      <div className="sidebar__footer">
        <p>Local agent workspace</p>
        <span>行程、交通、住宿、餐饮与预算</span>
      </div>
    </aside>
  )
}
