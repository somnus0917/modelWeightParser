use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    widgets::{Block, Borders, Paragraph},
};
use std::io::{Result, stdout};
mod ui;
fn main() -> Result<()> {
    enable_raw_mode()?;
    // 进入备用屏幕（类似打开 vim，退出后终端恢复你原来的命令行历史）
    stdout().execute(EnterAlternateScreen)?;

    // 创建终端实例
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    loop {
        // 1. 绘制 UI
        terminal.draw(|frame| {
            // 创建一个段落组件 (Paragraph)，外加一个带有全边框的块 (Block)
            ui::draw(frame);
        })?;

        // 2. 处理键盘输入
        // 轮询事件，设置 50ms 超时，防止程序完全阻塞在等待输入上
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                // 如果按下的是 'q' 键，就跳出循环
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}
