# Amiri (classical Naskh) is not part of the Gotenberg image; take it from Ubuntu.
FROM ubuntu:24.04 AS fonts
RUN apt-get update && apt-get install -y --no-install-recommends fonts-hosny-amiri \
    && rm -rf /var/lib/apt/lists/*

FROM gotenberg/gotenberg:8.37.0
USER root
COPY --from=fonts /usr/share/fonts/opentype/fonts-hosny-amiri /usr/share/fonts/opentype/amiri
# Aliases for the Windows / Office Arabic fonts Word documents usually ask for
COPY 99-arabic-aliases.conf /etc/fonts/conf.d/99-arabic-aliases.conf
RUN fc-cache -f && fc-list | grep -c -i -E "amiri|naskh"
USER gotenberg
