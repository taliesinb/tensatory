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
