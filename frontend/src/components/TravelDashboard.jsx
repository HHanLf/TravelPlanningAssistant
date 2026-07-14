import { motion } from 'framer-motion'
import { Calendar, CloudSun, ExternalLink, Hotel, Map as MapIcon, MapPin, Navigation, NotebookText, Star, Ticket, Utensils } from 'lucide-react'

const FALLBACK_IMAGES = {
  attraction: [
    'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1528127269322-539801943592?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1493246507139-91e8fad9978e?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=900&q=80',
  ],
  hotel: [
    'https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1578683010236-d716f9a3f461?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1590490360182-c33d57733427?auto=format&fit=crop&w=900&q=80',
  ],
  restaurant: [
    'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1551218808-94e220e084d2?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1552566626-52f8b828add9?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1544148103-0773bf10d330?auto=format&fit=crop&w=900&q=80',
  ],
  social: [
    'https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1496950866446-3253e1470e8e?auto=format&fit=crop&w=900&q=80',
    'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?auto=format&fit=crop&w=900&q=80',
  ],
}

function fallbackImage(seed = '', category = 'attraction') {
  const images = FALLBACK_IMAGES[category] || FALLBACK_IMAGES.attraction
  const text = String(seed || category)
  let hash = 0
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0
  }
  return images[hash % images.length]
}

function displayImage(item, category, seed) {
  const raw = item?.raw || item || {}
  return item?.image || item?.cover || raw.image || raw.cover || fallbackImage(seed || normalizeTitle(item, category), category)
}

function pickItems(state, key, toolName, payloadKey) {
  const fromKnowledge = state?.knowledge_summary?.[key]
  if (Array.isArray(fromKnowledge) && fromKnowledge.length) return fromKnowledge
  const result = state?.tool_results?.by_name?.[toolName]?.[0] || state?.tool_results?.[toolName]
  const payload = result?.payload || {}
  return Array.isArray(payload[payloadKey]) ? payload[payloadKey] : []
}

function unwrapToolPayload(result) {
  if (Array.isArray(result)) return unwrapToolPayload(result[0])
  if (!result || typeof result !== 'object') return {}
  if (result.payload && typeof result.payload === 'object') return result.payload
  if (result.result && typeof result.result === 'object') return result.result
  if (result.data && typeof result.data === 'object') return result.data
  return result
}

function findToolResult(state, names) {
  const toolResults = state?.tool_results
  if (!toolResults) return {}

  for (const name of names) {
    const byNameResult = toolResults?.by_name?.[name]?.[0]
    if (byNameResult) return unwrapToolPayload(byNameResult)

    const directResult = toolResults?.[name]
    if (directResult) return unwrapToolPayload(directResult)
  }

  const collections = [
    ...(Array.isArray(toolResults?.items) ? toolResults.items : []),
    ...(Array.isArray(toolResults?.results) ? toolResults.results : []),
    ...(Array.isArray(toolResults) ? toolResults : []),
  ]
  const matched = collections.find((item) => names.includes(item?.name) || names.includes(item?.tool))
  return unwrapToolPayload(matched)
}

function getWeather(state) {
  const weather = state?.knowledge_summary?.weather
  if (weather && Object.keys(weather).length) return weather
  return findToolResult(state, ['weather_lookup', 'weather'])
}

function normalizeTitle(item, fallback) {
  return item?.title || item?.name || item?.raw?.name || fallback
}

function normalizeContent(item) {
  return item?.content || item?.summary || item?.raw?.summary || item?.address || item?.raw?.address || ''
}

const CHINESE_DAY_NUMBERS = {
  一: 1,
  二: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
  十: 10,
}

function parseChineseDayNumber(value = '') {
  const text = String(value).trim()
  if (!text) return null
  if (CHINESE_DAY_NUMBERS[text]) return CHINESE_DAY_NUMBERS[text]
  if (text.length === 2 && text.startsWith('十')) return 10 + (CHINESE_DAY_NUMBERS[text[1]] || 0)
  if (text.length === 2 && text.endsWith('十')) return (CHINESE_DAY_NUMBERS[text[0]] || 1) * 10
  if (text.length === 3 && text[1] === '十') {
    return (CHINESE_DAY_NUMBERS[text[0]] || 0) * 10 + (CHINESE_DAY_NUMBERS[text[2]] || 0)
  }
  return null
}

function extractItineraryItems(dayText = '') {
  const lines = String(dayText || '')
    .split('\n')
    .map((line) => line.trim().replace(/^[\-*•]\s*/, ''))
    .filter(Boolean)
  const items = []

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^(上午|中午|午餐|下午|傍晚|晚上|晚餐)\s*[:：]\s*(.+)$/)
    if (!match) continue

    const details = []
    for (let nextIndex = index + 1; nextIndex < lines.length; nextIndex += 1) {
      if (/^(上午|中午|午餐|下午|傍晚|晚上|晚餐)\s*[:：]/.test(lines[nextIndex])) break
      if (/^(住宿建议|交通建议|预算|提醒|总结|小贴士)/.test(lines[nextIndex])) break
      details.push(lines[nextIndex])
    }

    const content =
      details.find((line) => line.startsWith('亮点')) ||
      details.find((line) => line.startsWith('建议')) ||
      details[0] ||
      '按当天节奏顺路安排，出发前再确认开放时间和交通。'

    items.push({
      title: `${match[1]}：${match[2]}`,
      content: content.replace(/^亮点\s*[:：]\s*/, '').replace(/^建议游玩时间\s*[:：]\s*/, '建议游玩时间：'),
    })
  }

  return items.slice(0, 4)
}

function splitDays(answer = '', days = 3) {
  const text = String(answer || '')
  const dayCount = Math.min(Math.max(Number(days) || 3, 1), 7)
  const fallbackSummaries = Array.from({ length: dayCount }, (_, index) => ({
    title: `Day ${index + 1}`,
    summary: index === 0 ? '抵达与城市初体验' : index === dayCount - 1 ? '轻松收尾与返程缓冲' : '核心片区深度游',
    items: [],
  }))
  const pattern = /(?:^|\n)\s*(?:[-*]\s*)?(?:(?:Day)\s*(\d+)|第\s*(\d+)\s*天|第([一二三四五六七八九十]+)天)\s*[:：.、-]?\s*([^\n]*)/gi
  const byDay = new Map()
  const matches = [...text.matchAll(pattern)]

  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index]
    const dayNumber = Number(match[1] || match[2]) || parseChineseDayNumber(match[3])
    if (!dayNumber || dayNumber < 1 || dayNumber > dayCount || byDay.has(dayNumber)) continue

    const blockStart = (match.index || 0) + match[0].length
    const blockEnd = index + 1 < matches.length ? matches[index + 1].index : text.length
    byDay.set(dayNumber, {
      summary: (match[4] || '').trim(),
      items: extractItineraryItems(text.slice(blockStart, blockEnd)),
    })
  }

  if (!byDay.size) return fallbackSummaries

  return fallbackSummaries.map((fallback, index) => ({
    title: `Day ${index + 1}`,
    summary: byDay.get(index + 1)?.summary || fallback.summary,
    items: byDay.get(index + 1)?.items || [],
  }))
}

function SectionHeader({ icon: Icon, eyebrow, title }) {
  return (
    <div className="section-heading">
      <div>
        <p>{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      <Icon size={18} />
    </div>
  )
}

function TimelineCard({ day, attractions }) {
  const attractionSpots = attractions.slice(day.index * 2, day.index * 2 + 2)
  const spots = attractionSpots.length ? attractionSpots : day.items || []
  return (
    <motion.article className="timeline-card" whileHover={{ y: -2 }} transition={{ duration: 0.18 }}>
      <div className="timeline-card__day">{day.title}</div>
      <div>
        <h3>{day.summary}</h3>
        <div className="spot-list">
          {(spots.length ? spots : [{ title: '核心景点待研究', content: 'Agent 会根据工具结果补充地址、时间和交通建议。' }]).map((spot, index) => (
            <div className="spot-row" key={`${normalizeTitle(spot, 'spot')}-${index}`}>
              <img src={displayImage(spot, 'attraction', `${day.title}-${normalizeTitle(spot, index)}`)} alt="" />
              <div>
                <strong>{normalizeTitle(spot, '旅行点位')}</strong>
                <span>{normalizeContent(spot) || '建议预留 2-3 小时，按同片区顺路串联。'}</span>
                <div className="metadata-row">
                  <span>
                    <MapPin size={12} /> {spot.raw?.address || spot.address || '地址待确认'}
                  </span>
                  <span>
                    <Star size={12} /> {spot.raw?.rating || spot.confidence || '推荐'}
                  </span>
                  <span>
                    <Ticket size={12} /> {spot.raw?.ticket || '门票待确认'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.article>
  )
}

function MapCard({ attractions, hotels, restaurants }) {
  return (
    <section className="dashboard-card map-card">
      <SectionHeader icon={MapIcon} eyebrow="Map" title="路线地图" />
      <div className="map-placeholder">
        <div className="map-grid" />
        <div className="map-pin map-pin--one" />
        <div className="map-pin map-pin--two" />
        <div className="map-pin map-pin--three" />
        <div className="map-route" />
        <div className="map-note">
          <Navigation size={16} />
          <span>地图组件预留：{attractions.length} 个景点 · {hotels.length} 家酒店 · {restaurants.length} 家餐厅</span>
        </div>
      </div>
    </section>
  )
}

function WeatherCard({ weather }) {
  const raw = unwrapToolPayload(weather?.raw || weather)
  const low = raw.temperature_min ?? raw.low ?? raw.min_temperature ?? raw.current_temperature ?? '--'
  const high = raw.temperature_max ?? raw.high ?? raw.max_temperature ?? '--'
  raw.temperature_min = low === '--' ? '' : String(low)
  raw.temperature_max = high === '--' ? '' : String(high)
  return (
    <section className="dashboard-card weather-card">
      <SectionHeader icon={CloudSun} eyebrow="Weather" title="天气与出行" />
      <div className="weather-main">
        <strong>{raw.forecast || weather?.summary || '天气待查询'}</strong>
        <span>{raw.temperature_min || raw.current_temperature || '--'}° / {raw.temperature_max || '--'}°</span>
      </div>
      <div className="info-grid">
        <div>
          <span>空气质量</span>
          <b>{raw.air_quality || raw.aqi || '待确认'}</b>
        </div>
        <div>
          <span>穿衣建议</span>
          <b>{raw.recommendation || '出发前复核实时天气'}</b>
        </div>
      </div>
    </section>
  )
}

function HotelCard({ item }) {
  const raw = item.raw || item
  return (
    <motion.article className="mini-card" whileHover={{ y: -2 }}>
      <img src={displayImage(item, 'hotel', normalizeTitle(item, 'hotel'))} alt="" />
      <div>
        <strong>{normalizeTitle(item, '酒店候选')}</strong>
        <p>{raw.area || raw.location || item.content || '适合作为行程落点'}</p>
        <div className="metadata-row">
          <span>¥{raw.price || '待查'}</span>
          <span>评分 {raw.rating || item.confidence || '-'}</span>
          <span>{raw.distance_hint || '距离待确认'}</span>
        </div>
      </div>
    </motion.article>
  )
}

function RestaurantCard({ item }) {
  const raw = item.raw || item
  return (
    <motion.article className="mini-card" whileHover={{ y: -2 }}>
      <img src={displayImage(item, 'restaurant', normalizeTitle(item, 'restaurant'))} alt="" />
      <div>
        <strong>{normalizeTitle(item, '餐厅候选')}</strong>
        <p>{normalizeContent(item) || '适合放在当天片区附近解决晚餐'}</p>
        <div className="metadata-row">
          <span>评分 {raw.rating || item.confidence || '-'}</span>
          <span>{raw.recommended_dish || raw.category || '推荐菜待补充'}</span>
          <span>{raw.distance || '距离待确认'}</span>
        </div>
      </div>
    </motion.article>
  )
}

function XiaohongshuCard({ item }) {
  const raw = item.raw || item
  return (
    <article className="xhs-card">
      <img src={displayImage(item, 'social', normalizeTitle(item, 'social'))} alt="" />
      <div>
        <strong>{normalizeTitle(item, '小红书经验')}</strong>
        <p>{normalizeContent(item) || raw.insight || '等待更多本地经验总结'}</p>
        <div className="xhs-card__footer">
          <span>点赞 {raw.liked_count || '-'}</span>
          {raw.url ? (
            <a href={raw.url} target="_blank" rel="noreferrer">
              查看 <ExternalLink size={12} />
            </a>
          ) : (
            <span>来源摘要</span>
          )}
        </div>
      </div>
    </article>
  )
}

export function TravelDashboard({ state = {}, loading }) {
  const problem = state.problem || {}
  const profile = state.profile || state.memory_context?.user_profile || {}
  const destination = problem.destination || profile.destination || state.knowledge_summary?.destination || '目的地'
  const days = Number(problem.days || profile.days || 3)
  const attractions = pickItems(state, 'attractions', 'place_search', 'places')
  const hotels = pickItems(state, 'hotels', 'hotel_search', 'hotels')
  const restaurants = pickItems(state, 'restaurants', 'restaurant_recommendation', 'places')
  const xhs = [
    ...(state.knowledge_summary?.xiaohongshu_insights || []).map((content, index) => ({ title: `经验 ${index + 1}`, content })),
    ...pickItems(state, 'social', 'xiaohongshu_search', 'notes'),
  ].slice(0, 5)
  const weather = getWeather(state)
  const timeline = splitDays(state.answer || state.final_answer, days).map((item, index) => ({ ...item, index }))

  return (
    <aside className="dashboard">
      <div className="dashboard__header">
        <div>
          <p>Travel Dashboard</p>
          <h2>{destination} 旅行看板</h2>
        </div>
        <span>{loading ? '生成中' : '实时预览'}</span>
      </div>

      <div className="dashboard__scroll">
        <section className="dashboard-card">
          <SectionHeader icon={Calendar} eyebrow="Timeline" title="行程 Timeline" />
          <div className="timeline-list">
            {timeline.map((day) => (
              <TimelineCard key={day.title} day={day} attractions={attractions} />
            ))}
          </div>
        </section>

        <MapCard attractions={attractions} hotels={hotels} restaurants={restaurants} />
        <WeatherCard weather={weather} />

        <section className="dashboard-card">
          <SectionHeader icon={Hotel} eyebrow="Stay" title="酒店推荐" />
          <div className="mini-card-list">
            {(hotels.length ? hotels : [{ title: '酒店候选待生成', content: '建议优先靠近地铁或核心片区，降低每日通勤成本。' }]).slice(0, 4).map((item, index) => (
              <HotelCard item={item} key={`${normalizeTitle(item, 'hotel')}-${index}`} />
            ))}
          </div>
        </section>

        <section className="dashboard-card">
          <SectionHeader icon={Utensils} eyebrow="Food" title="餐厅推荐" />
          <div className="mini-card-list">
            {(restaurants.length ? restaurants : [{ title: '餐厅候选待生成', content: '晚餐优先放在当天片区附近，不为单店跨城折返。' }]).slice(0, 4).map((item, index) => (
              <RestaurantCard item={item} key={`${normalizeTitle(item, 'restaurant')}-${index}`} />
            ))}
          </div>
        </section>

        <section className="dashboard-card">
          <SectionHeader icon={NotebookText} eyebrow="Xiaohongshu" title="小红书经验" />
          <div className="xhs-list">
            {(xhs.length ? xhs : [{ title: '本地经验待总结', content: '这里会展示避坑、排队、预约、机位和区域建议。' }]).map((item, index) => (
              <XiaohongshuCard item={item} key={`${normalizeTitle(item, 'xhs')}-${index}`} />
            ))}
          </div>
        </section>
      </div>
    </aside>
  )
}
