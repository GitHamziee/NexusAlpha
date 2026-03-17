FROM freqtradeorg/freqtrade:stable

USER root

# Copy and install Python dependencies (lean — no ML in Phase 1)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER ftuser

# Copy strategy files
COPY --chown=ftuser:ftuser strategies/ /freqtrade/strategies/
COPY --chown=ftuser:ftuser config/ /freqtrade/config/
COPY --chown=ftuser:ftuser scripts/ /freqtrade/scripts/
COPY --chown=ftuser:ftuser models/ /freqtrade/models/

WORKDIR /freqtrade
