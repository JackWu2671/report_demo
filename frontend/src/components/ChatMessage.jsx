import React from 'react'

const UserAvatar = () => (
  <div className="msg-avatar msg-avatar--user">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
    </svg>
  </div>
)

const AssistantAvatar = () => (
  <div className="msg-avatar msg-avatar--assistant">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
      <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-9 9l-5 3V9l5 3z"/>
    </svg>
  </div>
)

function extractChat(text) {
  const idx = text.indexOf('---OUTLINE---')
  return idx >= 0 ? text.slice(0, idx).trim() : text
}

export default function ChatMessage({ message, isStreaming }) {
  const isUser = message.role === 'user'
  const content = isUser ? message.content : extractChat(message.content)

  return (
    <div className={`msg-row msg-row--${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <AssistantAvatar />}
      <div className="msg-bubble-wrap">
        <div className={`msg-bubble msg-bubble--${isUser ? 'user' : 'assistant'}`}>
          {content || (
            !isUser && isStreaming
              ? <span className="msg-typing"><span /><span /><span /></span>
              : '…'
          )}
        </div>
      </div>
      {isUser && <UserAvatar />}
    </div>
  )
}
