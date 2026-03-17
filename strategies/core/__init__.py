# Core strategy modules — imported after each is built
from strategies.core.regime_detector import (  # noqa: F401
    add_regime_indicators,
    apply_regime,
    classify_regime,
    confirm_regime_multitf,
)

from strategies.core.trend_following import (  # noqa: F401
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)
from strategies.core.mean_reversion import (  # noqa: F401
    add_mr_indicators,
    populate_mr_entries,
    populate_mr_exits,
)
from strategies.core.funding_rate import (  # noqa: F401
    add_funding_indicators,
    populate_funding_entries,
    populate_funding_exits,
)
