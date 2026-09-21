/** Base class for all errors raised while parsing or evaluating a bundle. */
export class TensatoryError extends Error {
  constructor(message: string, readonly path: string[] = []) {
    super(path.length ? `${path.join(".")}: ${message}` : message);
    this.name = new.target.name;
  }
}

/** The spec is structurally valid but semantically wrong (shape mismatch, unknown name, ...). */
export class SpecError extends TensatoryError {}

/** A feature that is described by the schema but not implemented (yet). */
export class NotSupportedError extends TensatoryError {}

/** A runtime evaluation problem (point outside box, ...). */
export class EvalError extends TensatoryError {}

export function assertSpec(cond: unknown, message: string, path: string[] = []): asserts cond {
  if (!cond) throw new SpecError(message, path);
}

/** a zod issue, as far as we read it: `invalid_union` issues carry each branch's own issues */
export interface ZodIssueLike { message: string; path: readonly PropertyKey[]; code?: string; errors?: readonly (readonly ZodIssueLike[])[] }

/**
 * The most specific issue of a failed parse: descends into union failures, taking the branch whose issue lies
 * deepest (an inline bundle inside `string | BundleSpec` reports its missing `fields`, not "invalid input").
 */
export function specificIssue(issues: readonly ZodIssueLike[]): ZodIssueLike {
  let best = issues[0]!;
  for (const issue of issues) {
    if (issue.code === "invalid_union" && issue.errors?.length) {
      for (const branch of issue.errors) {
        if (!branch.length) continue;
        const inner = specificIssue(branch);
        if (inner.path.length > best.path.length || best.code === "invalid_union") best = inner;
      }
    }
  }
  return best;
}

/** the SpecError for a failed zod parse */
export function specErrorOf(issues: readonly ZodIssueLike[]): SpecError {
  const issue = specificIssue(issues);
  const more = issues.length > 1 ? ` (+${issues.length - 1} more issues)` : "";
  return new SpecError(`${issue.message}${more}`, issue.path.map(String));
}
