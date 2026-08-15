function backtest(data, parameters):

    equity = initial_capital
    trade_list = []

    for each candle in data:

        signal = generate_signal(data, parameters)

        if signal == BUY:
            open_long_position()

        if signal == SELL:
            open_short_position()

        update_equity()

        if stop_loss or take_profit_hit:
            close_position()
            record_trade()

    return trade_list