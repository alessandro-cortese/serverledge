function intParam(params, name, defaultValue, min, max) {
    const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
    const value = Number.isFinite(parsed) ? parsed : defaultValue;
    return Math.max(min, Math.min(max, value));
}

module.exports = (params, ctx) => {
    params = params || {};

    const iterations = intParam(
        params,
        "iterations",
        6000000,
        100000,
        20000000
    );

    let acc = 0.0;
    let x = 0.123456789;

    for (let i = 1; i <= iterations; i++) {
        x = (
            x * 1.0000001192092896
            + 0.00000035762786865
        ) % 1.0;

        acc +=
            Math.sin(
                x + (i % 31) * 0.001
            )
            * Math.cos(x * 0.5);

        acc +=
            Math.sqrt(x + 1.0)
            / (1.0 + (i % 17));
    }

    return {
        benchmark: "float_ops",
        iterations,
        result: Number(acc.toFixed(9)),
    };
};