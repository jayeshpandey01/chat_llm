"use client";

import { useState, useMemo } from "react";
import {
  CLINICAL_CATEGORIES,
  BODY_REGIONS,
  TOP_FREQUENT_CONDITIONS,
  CLINICAL_CLASSES,
  ClinicalClassItem,
} from "@/lib/clinical-classes";
import {
  Search,
  Plus,
  Check,
  ChevronDown,
  ChevronUp,
  Layers,
  Filter,
  Sparkles,
  MapPin,
  BookOpen,
  X,
  Compass,
} from "lucide-react";

interface ClinicalClassesBrowserProps {
  onLoadSymptoms: (conditionName: string, symptoms: string[], tier: "primary_41" | "general_50" | "all_580") => void;
  selectedTier: "auto" | "primary_41" | "general_50" | "all_580";
  onSelectTier: (tier: "auto" | "primary_41" | "general_50" | "all_580") => void;
  activeConditionName?: string | null;
  onClearCondition?: () => void;
}

export default function ClinicalClassesBrowser({
  onLoadSymptoms,
  selectedTier,
  onSelectTier,
  activeConditionName,
  onClearCondition,
}: ClinicalClassesBrowserProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [viewMode, setViewMode] = useState<"region" | "specialty" | "az">("region");
  const [selectedRegion, setSelectedRegion] = useState<string>("All Regions");
  const [selectedCategory, setSelectedCategory] = useState<string>("All Specialties");
  const [selectedLetter, setSelectedLetter] = useState<string>("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [loadedId, setLoadedId] = useState<string | null>(null);

  // Alphabetical counts for A-Z index bar
  const alphabetCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (let i = 65; i <= 90; i++) {
      counts[String.fromCharCode(i)] = 0;
    }
    CLINICAL_CLASSES.forEach((c) => {
      const firstChar = c.name.charAt(0).toUpperCase();
      if (counts[firstChar] !== undefined) {
        counts[firstChar]++;
      }
    });
    return counts;
  }, []);

  // Quick Curated Frequent Presets
  const frequentItems = useMemo(() => {
    const set = new Set<string>(TOP_FREQUENT_CONDITIONS);
    return CLINICAL_CLASSES.filter((c) => set.has(c.name) || set.has(c.common_name));
  }, []);

  // Filter classes across Multi-field Search, Tiers, and Active View Mode
  const filteredClasses = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();

    return CLINICAL_CLASSES.filter((item) => {
      // 1. Search Query (Multi-field match across name, common_name, category, body_region, synonyms, symptoms)
      if (q) {
        const matchesName = item.name.toLowerCase().includes(q);
        const matchesCommon = item.common_name?.toLowerCase().includes(q);
        const matchesCategory = item.category?.toLowerCase().includes(q);
        const matchesRegion = item.body_region?.toLowerCase().includes(q);
        const matchesSynonym = item.synonyms?.some((s) => s.toLowerCase().includes(q));
        const matchesSymptom = item.typical_symptoms?.some((s) => s.toLowerCase().includes(q));

        if (!matchesName && !matchesCommon && !matchesCategory && !matchesRegion && !matchesSynonym && !matchesSymptom) {
          return false;
        }
      }

      // 2. Tier Filter
      if (tierFilter !== "all" && item.tier !== tierFilter) {
        return false;
      }

      // 3. View Mode Specific Navigation Filters
      if (viewMode === "region") {
        if (selectedRegion !== "All Regions" && item.body_region !== selectedRegion) {
          return false;
        }
      } else if (viewMode === "specialty") {
        if (selectedCategory !== "All Specialties" && item.category !== selectedCategory) {
          return false;
        }
      } else if (viewMode === "az") {
        if (selectedLetter !== "All") {
          const firstChar = item.name.charAt(0).toUpperCase();
          if (firstChar !== selectedLetter) {
            return false;
          }
        }
      }

      return true;
    });
  }, [searchQuery, tierFilter, viewMode, selectedRegion, selectedCategory, selectedLetter]);

  const handleLoad = (item: ClinicalClassItem) => {
    const isCurrentlyActive =
      (activeConditionName && activeConditionName.toLowerCase() === item.name.toLowerCase()) ||
      loadedId === item.id;

    if (isCurrentlyActive && onClearCondition) {
      onClearCondition();
      setLoadedId(null);
      return;
    }

    onLoadSymptoms(item.name, item.typical_symptoms, item.tier);
    setLoadedId(item.id);
  };

  const clearAllFilters = () => {
    setSearchQuery("");
    setTierFilter("all");
    setSelectedRegion("All Regions");
    setSelectedCategory("All Specialties");
    setSelectedLetter("All");
  };

  return (
    <div className="border-2 border-[var(--border-color)] bg-white text-black shadow-[3px_3px_0px_var(--shadow-color)] mt-3">
      {/* Header Bar */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="p-3 bg-[var(--accent-color)] text-[var(--accent-text)] cursor-pointer flex items-center justify-between border-b-2 border-black select-none"
      >
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 stroke-[2.5]" />
          <h3 className="font-['Space_Grotesk'] text-xs font-extrabold uppercase tracking-tight">
            Arranged Clinical Classes Directory (All 580 Conditions)
          </h3>
          <span className="neo-badge neo-badge-green text-[9px]">
            {filteredClasses.length} / {CLINICAL_CLASSES.length} Ready
          </span>
          {activeConditionName && (
            <span className="neo-badge neo-badge-yellow text-[9px] hidden sm:inline-flex items-center gap-1 font-bold">
              <Check className="w-2.5 h-2.5 stroke-[3]" /> Active: {activeConditionName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono opacity-80 hidden sm:inline">
            {isOpen ? "Click to collapse" : "Click to expand & browse all classes"}
          </span>
          <button type="button" className="p-1 hover:opacity-75" aria-label="Toggle classes directory">
            {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="p-3.5 space-y-3 bg-[#fdfdfc]">
          {/* Quick-Access Top 20 Curated Frequent Presets */}
          <div className="p-2.5 border border-black bg-amber-50/70 shadow-[1px_1px_0px_#000]">
            <div className="flex items-center justify-between gap-1 mb-1.5">
              <span className="font-['Space_Grotesk'] text-[10px] font-bold uppercase flex items-center gap-1 text-amber-950">
                <Sparkles className="w-3.5 h-3.5 text-amber-700 stroke-[2.5]" />
                Top Frequent Primary Care Presets (1-Click Load):
              </span>
              <span className="text-[9px] font-mono opacity-70">
                {frequentItems.length} Everyday Conditions
              </span>
            </div>
            <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
              {frequentItems.map((item) => {
                const isSelected =
                  (activeConditionName && activeConditionName.toLowerCase() === item.name.toLowerCase()) ||
                  loadedId === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleLoad(item)}
                    className={`whitespace-nowrap px-2 py-0.5 text-[10px] font-['Space_Grotesk'] font-bold border transition-all inline-flex items-center gap-1 ${
                      isSelected
                        ? "bg-amber-300 border-black text-black shadow-[1px_1px_0px_#000]"
                        : "bg-white border-neutral-400 text-neutral-800 hover:border-black hover:bg-neutral-100"
                    }`}
                  >
                    {isSelected ? <Check className="w-3 h-3 stroke-[3]" /> : <Plus className="w-3 h-3" />}
                    <span>{item.common_name || item.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Controls: Search & Tier Filter */}
          <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 text-xs">
            <div className="sm:col-span-8 relative">
              <Search className="w-4 h-4 absolute left-3 top-2.5 opacity-50" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search across all 580 conditions, common names, or symptoms (e.g. heartburn, asthma, rash)..."
                className="w-full pl-9 pr-8 py-2 border-2 border-black bg-white text-black font-['Inter'] text-xs outline-none shadow-[2px_2px_0px_#000]"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2.5 top-2.5 opacity-60 hover:opacity-100 p-0.5"
                  title="Clear search"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <div className="sm:col-span-4 flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 opacity-70" />
              <select
                value={tierFilter}
                onChange={(e) => setTierFilter(e.target.value)}
                className="w-full py-2 px-2 border-2 border-black bg-white text-black text-xs font-['Space_Grotesk'] font-bold uppercase outline-none shadow-[2px_2px_0px_#000]"
              >
                <option value="all">All Tiers (580 Classes)</option>
                <option value="primary_41">Primary Care (41 Classes)</option>
                <option value="general_50">General Multi-System (50 Classes)</option>
                <option value="all_580">Broad Universe (489 Classes)</option>
              </select>
            </div>
          </div>

          {/* Navigation Mode Switcher: By Body Region | By Specialty | A–Z Index */}
          <div className="flex items-center justify-between border-t border-b border-neutral-300 py-1.5 gap-2 flex-wrap">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-['Space_Grotesk'] font-bold uppercase opacity-75">
                Browse Mode:
              </span>
              <div className="inline-flex border border-black shadow-[1px_1px_0px_#000]">
                <button
                  type="button"
                  onClick={() => setViewMode("region")}
                  className={`px-2 py-0.5 text-[10px] font-['Space_Grotesk'] font-bold uppercase transition-all inline-flex items-center gap-1 ${
                    viewMode === "region" ? "bg-black text-white" : "bg-white text-black hover:bg-neutral-100"
                  }`}
                >
                  <MapPin className="w-3 h-3" /> By Body Region (7)
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("specialty")}
                  className={`px-2 py-0.5 text-[10px] font-['Space_Grotesk'] font-bold uppercase border-l border-black transition-all inline-flex items-center gap-1 ${
                    viewMode === "specialty" ? "bg-black text-white" : "bg-white text-black hover:bg-neutral-100"
                  }`}
                >
                  <Layers className="w-3 h-3" /> By Specialty (15)
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("az")}
                  className={`px-2 py-0.5 text-[10px] font-['Space_Grotesk'] font-bold uppercase border-l border-black transition-all inline-flex items-center gap-1 ${
                    viewMode === "az" ? "bg-black text-white" : "bg-white text-black hover:bg-neutral-100"
                  }`}
                >
                  <BookOpen className="w-3 h-3" /> A–Z Index
                </button>
              </div>
            </div>

            {(searchQuery || tierFilter !== "all" || selectedRegion !== "All Regions" || selectedCategory !== "All Specialties" || selectedLetter !== "All") && (
              <button
                type="button"
                onClick={clearAllFilters}
                className="text-[10px] font-mono text-neutral-600 hover:text-black underline flex items-center gap-1"
              >
                <X className="w-3 h-3" /> Reset Filters
              </button>
            )}
          </div>

          {/* Mode 1: Anatomical Body Region Tabs */}
          {viewMode === "region" && (
            <div>
              <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin">
                {BODY_REGIONS.map((region) => {
                  const count = region === "All Regions"
                    ? CLINICAL_CLASSES.length
                    : CLINICAL_CLASSES.filter((c) => c.body_region === region).length;
                  const isSelected = selectedRegion === region;
                  return (
                    <button
                      key={region}
                      type="button"
                      onClick={() => setSelectedRegion(region)}
                      className={`whitespace-nowrap px-2.5 py-1 text-[10px] font-['Space_Grotesk'] font-bold uppercase border border-black transition-all ${
                        isSelected
                          ? "bg-black text-white shadow-[2px_2px_0px_rgba(0,0,0,0.3)]"
                          : "bg-white text-black hover:bg-neutral-100"
                      }`}
                    >
                      {region} ({count})
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Mode 2: Clinical Specialty Tabs */}
          {viewMode === "specialty" && (
            <div>
              <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin">
                {CLINICAL_CATEGORIES.map((cat) => {
                  const count = cat === "All Specialties"
                    ? CLINICAL_CLASSES.length
                    : CLINICAL_CLASSES.filter((c) => c.category === cat).length;
                  const isSelected = selectedCategory === cat;
                  return (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setSelectedCategory(cat)}
                      className={`whitespace-nowrap px-2.5 py-1 text-[10px] font-['Space_Grotesk'] font-bold uppercase border border-black transition-all ${
                        isSelected
                          ? "bg-black text-white shadow-[2px_2px_0px_rgba(0,0,0,0.3)]"
                          : "bg-white text-black hover:bg-neutral-100"
                      }`}
                    >
                      {cat} ({count})
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Mode 3: A–Z Alphabetical Jump Bar */}
          {viewMode === "az" && (
            <div className="flex flex-wrap gap-1 p-1 bg-neutral-100 border border-neutral-300">
              <button
                type="button"
                onClick={() => setSelectedLetter("All")}
                className={`px-2 py-0.5 text-[10px] font-mono font-bold border transition-all ${
                  selectedLetter === "All"
                    ? "bg-black text-white border-black"
                    : "bg-white text-black border-neutral-300 hover:bg-neutral-200"
                }`}
              >
                ALL ({CLINICAL_CLASSES.length})
              </button>
              {Object.keys(alphabetCounts).map((letter) => {
                const count = alphabetCounts[letter] || 0;
                const isSelected = selectedLetter === letter;
                const isDisabled = count === 0;
                return (
                  <button
                    key={letter}
                    type="button"
                    disabled={isDisabled}
                    onClick={() => setSelectedLetter(letter)}
                    className={`px-1.5 py-0.5 text-[10px] font-mono font-bold border transition-all ${
                      isDisabled
                        ? "opacity-30 bg-neutral-100 border-neutral-200 cursor-not-allowed"
                        : isSelected
                        ? "bg-black text-white border-black"
                        : "bg-white text-black border-neutral-300 hover:bg-neutral-200"
                    }`}
                    title={`${letter}: ${count} conditions`}
                  >
                    {letter} <span className="text-[8px] opacity-75">({count})</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Results Cards List (Viewport-fitted, scrollable) */}
          <div className="max-h-[320px] overflow-y-auto space-y-1.5 pr-1 border border-neutral-300 p-2 bg-neutral-50/60">
            {filteredClasses.length === 0 ? (
              <div className="p-6 text-center text-xs opacity-60 italic space-y-1">
                <div>No matching clinical conditions found.</div>
                <button
                  type="button"
                  onClick={clearAllFilters}
                  className="font-bold underline text-blue-700 not-italic inline-block mt-1"
                >
                  Clear search &amp; filters
                </button>
              </div>
            ) : (
              filteredClasses.map((item) => {
                const isLoaded =
                  (activeConditionName && activeConditionName.toLowerCase() === item.name.toLowerCase()) ||
                  loadedId === item.id;

                const tierLabel = item.tier === "primary_41"
                  ? "Primary (41)"
                  : item.tier === "general_50"
                  ? "General (50)"
                  : "Universe (580)";

                const tierBadge = item.tier === "primary_41"
                  ? "neo-badge-green"
                  : item.tier === "general_50"
                  ? "neo-badge-yellow"
                  : "bg-neutral-200 text-black";

                // Clean typical symptoms string for display
                const readableSymptoms = (item.typical_symptoms || [])
                  .slice(0, 4)
                  .map((s) => s.replace(/_/g, " "))
                  .join(", ");

                return (
                  <div
                    key={item.id}
                    className={`p-2.5 border-2 border-black flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 transition-all text-xs ${
                      isLoaded
                        ? "bg-[#fef9c3] border-l-8 border-l-amber-500 shadow-[3px_3px_0px_#000]"
                        : "bg-white hover:bg-neutral-50 shadow-[1px_1px_0px_#000]"
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                        <span className="font-['Space_Grotesk'] font-extrabold uppercase text-xs truncate">
                          {item.name}
                        </span>
                        {item.common_name && item.common_name !== item.name && (
                          <span className="text-[10px] font-bold text-amber-900 bg-amber-100/90 px-1.5 py-0.2 border border-amber-300">
                            Common: {item.common_name}
                          </span>
                        )}
                        <span className="text-[9px] font-mono opacity-60">
                          [{item.category}]
                        </span>
                        <span className="text-[9px] font-mono opacity-60">
                          • {item.body_region}
                        </span>
                        <span className={`neo-badge text-[9px] ${tierBadge}`}>
                          {tierLabel}
                        </span>
                        {isLoaded && (
                          <span className="neo-badge neo-badge-green text-[9px] inline-flex items-center gap-1 font-bold">
                            <Check className="w-2.5 h-2.5 stroke-[3]" /> Active in Form
                          </span>
                        )}
                      </div>

                      {readableSymptoms && (
                        <div className="text-[10px] opacity-75 font-['Inter'] truncate">
                          <span className="font-semibold text-neutral-600">Hallmark Symptoms:</span> {readableSymptoms}
                        </div>
                      )}
                    </div>

                    <div className="shrink-0 flex items-center gap-1 w-full sm:w-auto justify-end">
                      <button
                        type="button"
                        onClick={() => handleLoad(item)}
                        className={`neo-btn !py-1 !px-2.5 !text-[11px] font-['Space_Grotesk'] font-bold uppercase transition-all inline-flex items-center gap-1 ${
                          isLoaded
                            ? "bg-amber-300 border-2 border-black text-black shadow-[1px_1px_0px_#000]"
                            : "bg-white hover:bg-neutral-100 text-black border-2 border-black shadow-[2px_2px_0px_#000]"
                        }`}
                      >
                        {isLoaded ? (
                          <>
                            <Check className="w-3 h-3 stroke-[3]" /> Loaded (Click to Unload)
                          </>
                        ) : (
                          <>
                            <Plus className="w-3 h-3 stroke-[2.5]" /> + Load into Form
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
