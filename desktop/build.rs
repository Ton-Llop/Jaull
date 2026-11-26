fn main() {
    slint_build::compile("ui/app.slint").expect("Slint UI should compile");
    // The icon of the executable itself; the window's icon is set in app.slint.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icon.ico");
        res.set("ProductName", "Jaull");
        res.set("FileDescription", "Jaull desktop");
        res.compile().expect("the Windows resources (icon) should compile");
    }
}
