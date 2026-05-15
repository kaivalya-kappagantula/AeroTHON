from setuptools import setup, find_packages
 
package_name = "aerothon"
 
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/qr_scanner_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AeroTHON 2026 Team",
    maintainer_email="team@aerothon2026.local",
    description="AeroTHON 2026 Mission 2 autonomous flight nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "qr_scanner_node = aerothon.qr_scanner_node:main",
        ],
    },
)