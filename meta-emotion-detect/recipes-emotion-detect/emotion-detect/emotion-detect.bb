LICENSE = "CLOSED"
LIC_FIELS_CHKSUM = " "

SRC_URI = "file://app.py \
	file://haarcascade_frontalface_default.xml \
	file://em.tflite"

S = "${WORKDIR}"
TARGET_CC_ARCH += "${LDFLAGS}"

do_install () {
	install -d ${D}${bindir}
	install -m 0755 ${S}/app.py ${D}${bindir}
	install -m 0755 ${S}/haarcascade_frontalface_default.xml ${D}${bindir}
	install -m 0755 ${S}/em.tflite ${D}${bindir}
}
