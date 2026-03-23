// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PyCoreML",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "PyCoreML", type: .dynamic, targets: ["PyCoreML"]),
    ],
    dependencies: [
        .package(url: "https://github.com/jagtesh/ApplePy.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "PyCoreML",
            dependencies: [
                .product(name: "ApplePy", package: "ApplePy"),
                .product(name: "ApplePyClient", package: "ApplePy"),
            ],
            linkerSettings: [
                .linkedFramework("CoreML"),
            ]
        ),
    ]
)
