function intParam(params, name, defaultValue, min, max) {
    const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
    const value = Number.isFinite(parsed) ? parsed : defaultValue;
    return Math.max(min, Math.min(max, value));
}

module.exports = (params, ctx) => {
    params = params || {};

    const elements = intParam(
        params,
        "elements",
        3000000,
        100000,
        4000000
    );

    const rounds = intParam(
        params,
        "rounds",
        40,
        1,
        200
    );

    const data = new Uint32Array(elements);

    for (let i = 0; i < elements; i++) {
        data[i] =
            Math.imul(i, 0x9e3779b1) >>> 0;
    }

    let checksum = 0;

    for (let r = 0; r < rounds; r++) {
        const step = 97 + (r % 13);
        const offset = (r * 8191) % elements;

        for (
            let i = 0;
            i < elements;
            i += step
        ) {
            const idx =
                (i + offset) % elements;

            let value = data[idx];

            value = (
                value
                ^ Math.imul(
                    idx + r,
                    0x85ebca77
                )
            ) >>> 0;

            data[idx] = value;

            checksum = (
                checksum + value
            ) >>> 0;
        }

        let local = 0;

        for (
            let i = 0;
            i < elements;
            i += 4
        ) {
            local = (
                local + data[i]
            ) >>> 0;
        }

        checksum = (
            checksum ^ local
        ) >>> 0;
    }

    return {
        benchmark: "memory_rw",
        elements,
        rounds,
        bytes:
            elements
            * Uint32Array.BYTES_PER_ELEMENT,
        checksum,
    };
};