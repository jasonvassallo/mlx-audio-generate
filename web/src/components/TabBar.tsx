interface TabBarProps {
  active: string;
  tabs: { id: string; label: string }[];
  onChange: (id: string) => void;
}

export default function TabBar({ active, tabs, onChange }: TabBarProps) {
  return (
    <div className="flex border-b border-border">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex-1 px-3 py-2 text-xs font-medium uppercase tracking-wider
            transition-colors
            ${
              active === tab.id
                ? "border-b-2 border-accent text-text-primary"
                : "text-text-muted hover:text-text-secondary"
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
