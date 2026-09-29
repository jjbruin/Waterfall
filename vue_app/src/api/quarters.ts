/**
 * The reportable quarters, and which one a screen should open on — ASKED, never
 * hardcoded.
 *
 * WHY THIS EXISTS. Two screens seeded their quarter box with a pinned quarter
 * literal. A literal is correct for exactly one quarter and silently wrong
 * from the day the next one ends: Review Tracking would open on a stale quarter
 * and the One Pager batch would build a stale document, neither with anything on
 * screen saying so. The single-deal One Pager already resolved its quarter from
 * the server's `available_quarters`; this gives the screens that have no deal
 * loaded the same answer from the same authority.
 *
 * THE SERVER DECIDES WHAT "CURRENT" MEANS. `/api/portfolio-snapshot/quarters`
 * derives it from the calendar — a quarter is reportable once it has ENDED — so
 * the rule lives in one place rather than being re-implemented per screen. Its
 * `default` is the newest ended quarter.
 *
 * CACHED PER PAGE LOAD because several screens ask and the answer cannot change
 * within a session. A FAILED lookup is not cached: the next caller retries
 * rather than inheriting an empty list for the life of the tab.
 *
 * It returns '' when it cannot tell, never a guess. An empty quarter box is
 * visibly unanswered; a plausible wrong quarter is not.
 */
import api from './client'

export interface QuarterOptions {
  quarters: string[]
  default: string
}

let cached: Promise<QuarterOptions> | null = null

export function loadQuarters(): Promise<QuarterOptions> {
  if (!cached) {
    cached = api.get('/api/portfolio-snapshot/quarters')
      .then(r => ({
        quarters: (r.data?.quarters || []) as string[],
        default: (r.data?.default || '') as string,
      }))
      .catch(() => {
        cached = null                 // a failure must not become the answer
        return { quarters: [], default: '' }
      })
  }
  return cached
}

/** The quarter a screen should open on, or '' when the server cannot say. */
export async function defaultQuarter(): Promise<string> {
  return (await loadQuarters()).default
}
