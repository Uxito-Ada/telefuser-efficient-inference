document.querySelectorAll("[data-sync-group]").forEach((group) => {
  const players = [...group.querySelectorAll("video")];
  let syncing = false;

  const mirror = (source, action) => {
    if (syncing) return;
    syncing = true;
    players
      .filter((player) => player !== source)
      .forEach((player) => {
        if (Math.abs(player.currentTime - source.currentTime) > 0.08) {
          player.currentTime = source.currentTime;
        }
        action(player);
      });
    syncing = false;
  };

  players.forEach((player) => {
    player.addEventListener("play", () => {
      player.muted = false;
      mirror(player, (peer) => {
        peer.muted = true;
        peer.play().catch(() => {});
      });
    });
    player.addEventListener("pause", () => {
      mirror(player, (peer) => peer.pause());
    });
    player.addEventListener("seeked", () => {
      mirror(player, () => {});
    });
  });
});
