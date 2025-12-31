import { useState, useRef, useEffect } from "react";
import { CornerDownLeft, Command } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

interface TerminalPromptProps {
  onCommand: (command: string) => void;
  disabled?: boolean;
}

const SUGGESTIONS = [
  { cmd: "help", desc: "Show available commands" },
  { cmd: "status", desc: "System diagnostics" },
  { cmd: "build", desc: "Compile project" },
  { cmd: "deploy agent", desc: "Spawn autonomous agent" },
  { cmd: "connect dgx", desc: "Link external compute" },
  { cmd: "research", desc: "Initiate neural search" },
  { cmd: "clear", desc: "Clear terminal output" },
  { cmd: "/agents", desc: "List active agents" },
  { cmd: "/mcp", desc: "Show MCP connections" },
  { cmd: "/market", desc: "Browse plugin marketplace" },
  { cmd: "/settings", desc: "Open system configuration" },
  { cmd: "/plugins", desc: "Browse plugin marketplace" },
];

export function TerminalPrompt({ onCommand, disabled }: TerminalPromptProps) {
  const [input, setInput] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredSuggestions = SUGGESTIONS.filter(s => 
    s.cmd.toLowerCase().startsWith(input.toLowerCase()) && input.length > 0
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onCommand(input);
    setInput("");
    setShowSuggestions(false);
    setSelectedIndex(0);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showSuggestions && filteredSuggestions.length > 0) {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev > 0 ? prev - 1 : filteredSuggestions.length - 1));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev < filteredSuggestions.length - 1 ? prev + 1 : 0));
      } else if (e.key === 'Tab') {
        e.preventDefault();
        setInput(filteredSuggestions[selectedIndex].cmd);
        setShowSuggestions(false);
      }
    }
  };

  useEffect(() => {
    setShowSuggestions(filteredSuggestions.length > 0);
    setSelectedIndex(0);
  }, [input]);

  // Auto-focus logic
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  return (
    <div className="relative">
      <AnimatePresence>
        {showSuggestions && !disabled && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute bottom-full left-0 w-full mb-2 bg-black/90 border border-white/10 rounded-md overflow-hidden backdrop-blur-xl shadow-2xl z-50"
          >
            <div className="text-[10px] bg-white/5 px-2 py-1 text-muted-foreground border-b border-white/5 uppercase tracking-wider font-bold">
              Suggested Commands
            </div>
            {filteredSuggestions.map((s, i) => (
              <div
                key={s.cmd}
                onClick={() => {
                  setInput(s.cmd);
                  setShowSuggestions(false);
                  inputRef.current?.focus();
                }}
                className={`px-3 py-2 flex items-center justify-between cursor-pointer transition-colors text-xs font-mono ${
                  i === selectedIndex ? "bg-primary/20 text-primary" : "hover:bg-white/5 text-muted-foreground"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Command size={12} className={i === selectedIndex ? "opacity-100" : "opacity-50"} />
                  <span className={i === selectedIndex ? "font-bold" : ""}>{s.cmd}</span>
                </div>
                <span className="text-[10px] opacity-50">{s.desc}</span>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <form onSubmit={handleSubmit} className="flex items-center gap-2 p-2 bg-card/30 border-t border-border backdrop-blur-sm relative z-20">
        <span className="text-primary font-bold animate-pulse">{">"}</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className="flex-1 bg-transparent border-none outline-none text-foreground font-mono placeholder:text-muted-foreground/50"
          placeholder={disabled ? "PROCESSING..." : "Enter command..."}
          autoComplete="off"
          spellCheck="false"
        />
        <button 
          type="submit" 
          disabled={disabled || !input.trim()}
          className="text-primary/70 hover:text-primary transition-colors disabled:opacity-30"
        >
          <CornerDownLeft size={16} />
        </button>
      </form>
    </div>
  );
}
